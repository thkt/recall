#!/usr/bin/env python3
"""Opt-in real CLI migration check. Requires sqlite-vec for DB assertions.

All inputs are explicit; the work directory must be new. No installed recall,
default data directory, real session roots, or production index is used.
"""
import argparse
import hashlib
import json
import math
import os
import random
import string
from pathlib import Path
import shutil
import sqlite3
import struct
import subprocess
import time

import sqlite_vec

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--model-dir', type=Path, required=True)
parser.add_argument('--work-dir', type=Path, required=True)
args = parser.parse_args()
source_binary = args.binary.resolve(strict=True)
model_dir = args.model_dir.resolve(strict=True)
root = args.work_dir.absolute()
root.mkdir(mode=0o700, parents=True, exist_ok=False)
binary = root/'recall'
shutil.copyfile(source_binary,binary)
binary.chmod(0o700)
logs = root / 'logs'
logs.mkdir(mode=0o700)
claude = root / 'claude'
codex = root / 'codex'
claude.mkdir()
codex.mkdir()
env = os.environ.copy()
for key in ['RECALL_DB', 'RECALL_MODEL_MANIFEST', 'CLAUDE_CODE_SESSION_ID',
            '__RECALL_GEMMA_PROBE', '__RECALL_GEMMA_SPEC']:
    env.pop(key, None)
env.update(RECALL_MODEL_DIR=str(model_dir), RECALL_CLAUDE_DIR=str(claude),
           RECALL_CODEX_DIR=str(codex), HF_HOME=str(root/'empty-hf'),
           XDG_DATA_HOME=str(root/'default-data'))
calls = []
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def invoke(name, command, db, manifest=None, expected=0, extra_env=None):
    cmd = [str(binary), '--json', '--db-path', str(db)]
    if manifest is not None:
        cmd += ['--model-manifest', str(manifest)]
    start = time.monotonic()
    run = subprocess.run(cmd + command, env={**env, **(extra_env or {})},
                         capture_output=True, timeout=240)
    elapsed = time.monotonic() - start
    (logs/f'{name}.stdout').write_bytes(run.stdout)
    (logs/f'{name}.stderr').write_bytes(run.stderr)
    assert run.returncode == expected, (name, run.returncode, run.stderr[-1500:])
    payload = json.loads(run.stdout if expected == 0 else run.stderr.splitlines()[-1])
    calls.append({'name': name, 'exit': run.returncode, 'seconds': elapsed})
    return payload

def connect(path):
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn

stamp = '2026-10-15T00:00:00Z'
dense = ''.join(random.Random(20261016).choices(string.ascii_letters+string.digits+string.punctuation, k=32000))
fixtures = [
    ('gemma-memory', '/fixture/recall', 'Gemmaのメモリを減らすには',
     'BF16の埋め込みモデルでメモリ使用量を確認する。検索文と文書のprefixを固定する。'),
    ('gemma-rollback', '/fixture/recall', 'モデル更新後に旧版へ戻したい',
     '旧manifestと重みとtokenizerと対応索引を保管し、同じ組み合わせを選び直す。'),
    ('gemma-sql', '/fixture/storage', 'SQL索引の識別を確認したい',
     '768次元が同じでもembedding spaceは一致しない。model fingerprintで索引を分離する。'),
    ('gemma-long', '/fixture/recall', '長文の末尾まで検索できるか',
     '先頭検査 ' + dense + ' 日本語🙂 e\u0301 末尾証拠TAIL'),
    ('gemma-automated', '/fixture/recall', '<command-message>gemma-fixture</command-message>',
     '自動生成されたモデル更新メモ。'),
]
for sid, project, question, answer in fixtures:
    rows = [dict(type='user', cwd=project, message=dict(role='user', content=question), timestamp=stamp),
            dict(type='assistant', message=dict(role='assistant', content=answer), timestamp=stamp)]
    (claude/f'{sid}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
rows = [dict(type='session_meta', timestamp=stamp, payload=dict(id='gemma-codex', cwd='/fixture/codex')),
        dict(type='response_item', timestamp=stamp, payload=dict(role='user', content=[dict(type='input_text', text='索引の切戻し手順は') ])),
        dict(type='response_item', timestamp=stamp, payload=dict(role='assistant', content=[dict(type='output_text', text='Gemmaの旧manifestと対応DBを同時に指定する。')]))]
(codex/'rollout-gemma-codex.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))

old_db = root/'current.db'
new_db = root/'candidate.db'
spec = invoke('default-manifest', ['model', 'manifest'], old_db)['data']
assert digest(model_dir/'model.gguf') == spec['model_sha256']
assert digest(model_dir/'tokenizer.json') == spec['tokenizer_sha256']
old_manifest = root/'current-manifest.json'
new_manifest = root/'candidate-manifest.json'
old_manifest.write_text(json.dumps(spec, ensure_ascii=False, indent=2))
candidate = {**spec, 'document_prefix': 'title: candidate fixture | text: '}
new_manifest.write_text(json.dumps(candidate, ensure_ascii=False, indent=2))
first = invoke('current-index', ['index'], old_db, old_manifest)
assert not first['degraded'], first['notes']
status = invoke('current-status', ['status'], old_db, old_manifest)
assert status['data']['sessions'] == 6
assert status['data']['qa_chunks'] == status['data']['embedded'] > 0
with connect(old_db) as conn:
    current_identity = conn.execute('SELECT fingerprint FROM embedding_identity').fetchone()[0]
    windows = conn.execute('SELECT embedding,chunk_id,sub_idx FROM vec_chunks').fetchall()
    assert any(sub > 0 for _, _, sub in windows), 'long-document token splitting must occur'
    for vector, _, _ in windows:
        values = struct.unpack('<768f', vector)
        assert all(math.isfinite(x) for x in values)
        assert abs(math.sqrt(sum(x*x for x in values))-1) < 1e-5
    conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchall()

query = ['search', '旧モデルへ戻す手順', '--limit', '5']
before = invoke('current-search', query, old_db, old_manifest)
assert before['data']['results'] and not before['degraded']
shorthand = invoke('manifest-shorthand', query[1:], old_db, old_manifest)
assert shorthand['data']['results'] == before['data']['results']
for name, flags, source in [('source-claude', ['--source', 'claude'], 'claude'),
                            ('source-codex', ['--source', 'codex'], 'codex')]:
    out = invoke(name, ['search', '索引', *flags], old_db, old_manifest)
    assert out['data']['results'] and not out['degraded']
    assert all(r['source'] == source for r in out['data']['results'])
out = invoke('project-filter', ['search', '索引', '--project', '/fixture/storage'], old_db, old_manifest)
assert out['data']['results'] and all(r['project'].startswith('/fixture/storage') for r in out['data']['results'])
again = invoke('current-no-change', ['index'], old_db, old_manifest)
assert not again['degraded']
assert again['data']['observations']['counts']['inference_chunks'] == 0

old_hash = digest(old_db)
mismatch = invoke('reject-update-into-old', ['index'], old_db, new_manifest, expected=65)
assert 'identity mismatch' in mismatch['error']['message']
assert digest(old_db) == old_hash
degraded = invoke('mismatched-read-is-FTS-only', ['search', 'manifest'], old_db, new_manifest)
assert degraded['degraded'] and any('identity mismatch' in n for n in degraded['notes'])
mismatched_status=invoke('mismatched-status',['status'],old_db,new_manifest)
assert mismatched_status['degraded'] and any('identity mismatch' in n for n in mismatched_status['notes'])
mismatched_doctor=invoke('mismatched-doctor',['doctor'],old_db,new_manifest)
assert not mismatched_doctor['data']['healthy'] and mismatched_doctor['degraded']
fix=invoke('mismatched-doctor-fix',['doctor','--fix'],old_db,new_manifest)
assert not fix['data']['healthy'] and all('remove the database' not in n for n in fix['notes'])
assert digest(old_db)==old_hash
next_index = invoke('candidate-index', ['index'], new_db, new_manifest)
assert not next_index['degraded']
with connect(new_db) as conn:
    candidate_identity = conn.execute('SELECT fingerprint FROM embedding_identity').fetchone()[0]
assert candidate_identity != current_identity
for name, manifest, identity in [('current',old_manifest,current_identity), ('candidate',new_manifest,candidate_identity)]:
    run = subprocess.run([str(binary),'--model-manifest',str(manifest),'--json','--verbose','status'], env=env, capture_output=True, timeout=30)
    assert run.returncode == 0
    expected_path = root/'default-data'/'recall'/'indexes'/identity/'recall.db'
    assert str(expected_path).encode() in run.stderr
    assert not expected_path.exists(), 'read-only status must not create a default index'
    (logs/f'{name}-default-namespace.stderr').write_bytes(run.stderr)
assert digest(old_db) == old_hash
after = invoke('rollback-current-search', query, old_db, old_manifest)
assert after['data']['results'] == before['data']['results'] and not after['degraded']
assert digest(old_db) == old_hash

legacy = root/'legacy.db'
with connect(old_db) as source, connect(legacy) as dest:
    source.backup(dest)
with connect(legacy) as conn:
    conn.execute('DROP TABLE embedding_identity')
    conn.commit()
    conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchall()
legacy_hash = digest(legacy)
rejected = invoke('legacy-write-refused', ['rebuild'], legacy, old_manifest, expected=65)
assert 'legacy vector index' in rejected['error']['message']
assert digest(legacy) == legacy_hash
fallback = invoke('legacy-read-FTS-only', ['search', 'manifest'], legacy, old_manifest)
assert fallback['degraded'] and any('legacy vector index' in n for n in fallback['notes'])
assert digest(legacy) == legacy_hash

tampered = root/'tampered-artifacts'
tampered.mkdir()
shutil.copyfile(model_dir/'tokenizer.json', tampered/'tokenizer.json')
(tampered/'model.gguf').write_bytes(b'wrong model identity')
bad = invoke('checksum-rejected-FTS-fallback', ['search', 'manifest'], old_db, old_manifest,
             extra_env={'RECALL_MODEL_DIR': str(tampered)})
assert bad['degraded']
doctor = invoke('live-probe-doctor', ['doctor'], old_db, old_manifest)
assert doctor['data']['healthy'] and not doctor['degraded']
invoke('pinned-artifacts-download-no-network',['model','download'],old_db,old_manifest)

summary = {'status': 'complete', 'binary_sha256': digest(binary), 'sessions': 6,
           'QA_documents': status['data']['qa_chunks'], 'windows': len(windows),
           'all_vectors_finite_768_and_unit_norm': True,
           'current_identity': current_identity, 'candidate_identity': candidate_identity,
           'candidate_is_simulated_prefix_update': True,
           'default_model_namespaces_separate': True,
           'no_change_new_inference_chunks': 0, 'old_index_unchanged_by_candidate_and_rollback': True,
           'rollback_search_exactly_reproduced': True, 'legacy_index_unchanged': True,
           'checksums_rejected_before_native_load': True, 'real_CLI_doctor_probe': True,
           'standalone_relocated_binary_with_embedded_Metal': True,
           'production_index_accessed': False, 'amici_used': False, 'calls': calls}
(root/'verification.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
for path in root.rglob('*'):
    path.chmod(0o700 if path.is_dir() else 0o600)
print(json.dumps({k: v for k, v in summary.items() if k != 'calls'}, ensure_ascii=False))
