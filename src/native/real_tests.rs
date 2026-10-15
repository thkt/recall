//! Opt-in tests: inputs must be explicitly provided owned evaluation artifacts.
use super::*;
use crate::model::hash_file;
use serde_json::Value;
use std::{collections::HashMap, env, fs};

#[test]
#[ignore = "requires explicitly provided, verified public Gemma artifacts and the frozen independent vectors"]
fn real_model_boundaries_coverage_and_frozen_query_parity() {
    let root =
        env::var_os("RECALL_GEMMA_TEST_ARTIFACT_DIR").expect("explicit owned artifact directory");
    let root = Path::new(&root);
    let spec = ModelSpec::default();
    assert_eq!(
        hash_file(&root.join("model.gguf")).unwrap(),
        spec.model_sha256
    );
    assert_eq!(
        hash_file(&root.join("tokenizer.json")).unwrap(),
        spec.tokenizer_sha256
    );
    let gemma = Gemma::load(&root.join("model.gguf"), &root.join("tokenizer.json"), spec).unwrap();
    let mut checks = Vec::new();
    {
        let model = gemma.model.lock().unwrap();
        for target in [511, 512, 513, 8192, 8193] {
            let text = (target - 40..target + 1)
                .map(|n| "a ".repeat(n))
                .find(|s| {
                    model
                        .vocab()
                        .tokenize(
                            format!("{}{s}", gemma.spec.query_prefix).as_bytes(),
                            true,
                            false,
                        )
                        .len()
                        == target
                })
                .expect("construct exact tokenizer boundary");
            let t = gemma
                .query_tokens(&model, &text, &gemma.spec.query_prefix)
                .unwrap();
            assert_eq!(t.len(), target.min(MAX_TOKENS));
            if target > MAX_TOKENS {
                assert_eq!(t.last(), Some(&model.vocab().eos()));
            }
            let v = Gemma::encode(&model, &[t], &EmbedOptions::default()).unwrap();
            assert_eq!(v[0].len(), 768);
            checks.push(serde_json::json!({"original_tokens":target,"encoded_tokens":target.min(MAX_TOKENS),"passed":true}));
        }
        let doc = format!(
            "  {} 終端確認TAIL\n",
            "日本語🙂 e\u{301} token ".repeat(2500)
        );
        let windows = gemma.document_tokens(&model, &doc).unwrap();
        assert!(windows.len() > 1);
        let mut end = 0;
        for w in &windows {
            assert!(w.byte_start <= end && w.byte_end > w.byte_start);
            assert!(doc.is_char_boundary(w.byte_start) && doc.is_char_boundary(w.byte_end));
            assert!(w.tokens.len() <= MAX_TOKENS);
            let exact = model.vocab().tokenize(
                format!(
                    "{}{}",
                    gemma.spec.document_prefix,
                    &doc[w.byte_start..w.byte_end]
                )
                .as_bytes(),
                true,
                false,
            );
            assert_eq!(w.tokens, exact);
            end = end.max(w.byte_end);
        }
        assert_eq!(end, doc.len());
        checks.push(serde_json::json!({"long_document_UTF8_bytes":doc.len(),"windows":windows.len(),"all_bytes_covered":true,"all_retokenized_within_8192":true}));
    }
    let input = env::var_os("RECALL_GEMMA_TEST_QUERIES").expect("frozen independent query input");
    let vectors =
        env::var_os("RECALL_GEMMA_TEST_VECTORS").expect("frozen independent query vectors");
    let reference: HashMap<String, Value> = fs::read_to_string(vectors)
        .unwrap()
        .lines()
        .map(|l| {
            let v: Value = serde_json::from_str(l).unwrap();
            (v["id"].as_str().unwrap().to_owned(), v)
        })
        .collect();
    let mut count = 0;
    let mut max_error = 0_f64;
    for line in fs::read_to_string(input).unwrap().lines() {
        let q: Value = serde_json::from_str(line).unwrap();
        let id = q["id"].as_str().unwrap();
        let text = q["text"].as_str().unwrap();
        let actual = gemma.embed_query(text).unwrap();
        let expected = reference[id]["vectors"][0].as_array().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            max_error = max_error.max((f64::from(*a) - b.as_f64().unwrap()).abs());
        }
        count += 1;
    }
    assert_eq!(count, 146);
    assert!(max_error < 1e-5, "embedding contract drift: {max_error}");
    let result = serde_json::json!({"status":"complete","query_vectors_compared":count,"max_component_difference":max_error,"checks":checks});
    println!("{result}");
    if let Some(out) = env::var_os("RECALL_GEMMA_TEST_REPORT") {
        fs::write(out, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    }
}

#[test]
#[ignore = "requires the owned full-index copy and the frozen 286-request evaluation artifacts"]
fn frozen_full_index_search_contract_parity() {
    use crate::{
        db,
        embedding::ChunkedEmbedding,
        parser::Source,
        search::{self, CurrentSession, SearchOptions},
    };
    let fixture =
        env::var_os("RECALL_GEMMA_TEST_FULL_FIXTURE").expect("owned frozen evaluation directory");
    let fixture = Path::new(&fixture);
    let db_path = env::var_os("RECALL_GEMMA_TEST_DB").expect("explicit independent database copy");
    let (conn, _) = db::open_db_readonly(Path::new(&db_path)).unwrap();
    let query_rows = fs::read_to_string(fixture.join("queries.jsonl")).unwrap();
    let vectors: HashMap<String, Vec<f32>> =
        fs::read_to_string(fixture.join("gemma.query-vectors.jsonl"))
            .unwrap()
            .lines()
            .map(|s| {
                let r: Value = serde_json::from_str(s).unwrap();
                (
                    r["id"].as_str().unwrap().to_owned(),
                    r["vectors"][0]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| serde_json::from_value(x.clone()).unwrap())
                        .collect(),
                )
            })
            .collect();
    let expected: HashMap<(String, String), Value> =
        fs::read_to_string(fixture.join("gemma.quality-results.jsonl"))
            .unwrap()
            .lines()
            .map(|s| {
                let r: Value = serde_json::from_str(s).unwrap();
                (
                    (
                        r["id"].as_str().unwrap().to_owned(),
                        r["mode"].as_str().unwrap().to_owned(),
                    ),
                    r,
                )
            })
            .collect();
    struct Frozen(Vec<f32>);
    impl Embed for Frozen {
        fn embed_query(&self, _: &str) -> Result<Vec<f32>, EmbedError> {
            Ok(self.0.clone())
        }
        fn embed_document(&self, _: &str) -> Result<ChunkedEmbedding, EmbedError> {
            Err(EmbedError::EmptySequence)
        }
        fn embed_documents_batch(&self, _: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
            Err(EmbedError::EmptySequence)
        }
        fn embed_text(&self, _: &str, _: &str) -> Result<Vec<f32>, EmbedError> {
            Err(EmbedError::EmptySequence)
        }
    }
    let mut count = 0;
    let mut hits = 0;
    for s in query_rows.lines() {
        let q: Value = serde_json::from_str(s).unwrap();
        let id = q["id"].as_str().unwrap();
        let o = &q["options"];
        let string = |k: &str| o[k].as_str().map(str::to_owned);
        let current = if let Some(id) = o["current"]["exclude"].as_str() {
            CurrentSession::Exclude(id.to_owned())
        } else if let Some(id) = o["current"]["only"].as_str() {
            CurrentSession::Only(id.to_owned())
        } else {
            CurrentSession::Ignore
        };
        let opts = SearchOptions {
            project: string("project"),
            source: o["source"].as_str().and_then(Source::from_db),
            days: o["days"].as_i64(),
            file: string("file"),
            limit: o["limit"]
                .as_u64()
                .map(|x| usize::try_from(x).unwrap())
                .unwrap_or(10),
            current,
            include_automated: o["include_automated"].as_bool().unwrap_or(false),
            now_ms: Some(o["now_ms"].as_i64().unwrap_or(1790042494323)),
        };
        let embedder = Frozen(vectors[id].clone());
        for mode in ["fts", "hybrid"] {
            let model = if mode == "hybrid" {
                Some(&embedder as &dyn Embed)
            } else {
                None
            };
            let actual =
                search::search_with_embedder(&conn, q["text"].as_str(), &opts, model).unwrap();
            assert!(
                !actual.vec_degraded && !actual.results_incomplete,
                "id={id}, mode={mode}"
            );
            let result:Vec<Value>=actual.results.iter().enumerate().map(|(rank,r)|serde_json::json!({"session_id":r.session.session_id,"chunk_id":r.chunk_id,"excerpt":r.excerpt,"rank":rank+1})).collect();
            let reference = &expected[&(id.to_owned(), mode.to_owned())]["results"];
            assert_eq!(
                &serde_json::to_value(&result).unwrap(),
                reference,
                "full-index contract drift: {id} {mode}"
            );
            count += 1;
            hits += result.len();
        }
    }
    assert_eq!(count, 572);
    let result = serde_json::json!({"status":"complete","requests":286,"FTS_hybrid_rows_compared":count,"ranked_hits_compared":hits,"all_session_chunk_excerpt_equal":true,"queries_use_frozen_vectors":"native encoder parity tested separately"});
    println!("{result}");
    if let Some(path) = env::var_os("RECALL_GEMMA_TEST_SEARCH_REPORT") {
        fs::write(path, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    }
}
