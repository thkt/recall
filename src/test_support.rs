use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use rusqlite::Connection;

use crate::commands::search::SearchArgs;
use crate::embedding::Embed;
use crate::envelope::{CommandOutput, render_json_error, render_json_success};
use crate::error::{classify_exit_code, error_envelope};
use crate::model::DegradedReason;
use crate::{db, embedder};

pub(crate) fn vec_chunk_count(conn: &Connection) -> i64 {
    conn.query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap()
}

pub(crate) fn mock_loader() -> Result<Arc<dyn Embed>, DegradedReason> {
    Ok(Arc::new(embedder::MockEmbedder::new()) as Arc<dyn Embed>)
}

pub(crate) fn seed_healthy_index(conn: &Connection) {
    db::seed_session(conn, "s1");
    db::seed_chunk(conn, 1, "healthy chunk");
    let emb = embedder::MockEmbedder::deterministic_vector("healthy chunk");
    conn.execute(
        "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, 1, 0)",
        rusqlite::params![embedder::f32_as_bytes(&emb)],
    )
    .unwrap();
}

pub(crate) fn bare_empty_db(dir: &Path) -> PathBuf {
    let db_path = dir.join("recall.db");
    drop(rusqlite::Connection::open(&db_path).unwrap());
    db_path
}

pub(crate) fn search_command(query: &str) -> SearchArgs {
    SearchArgs {
        query: Some(query.to_owned()),
        project: None,
        days: None,
        source: None,
        file: None,
        limit: 10,
        exclude_current: false,
        include_current: false,
        only_current: false,
        include_automated: false,
    }
}

pub(crate) fn printed_envelope(out: &CommandOutput) -> serde_json::Value {
    serde_json::from_str(&render_json_success(out)).unwrap()
}

pub(crate) fn reported(err: &anyhow::Error) -> (ExitCode, String) {
    (
        classify_exit_code(err),
        render_json_error(&error_envelope(err)),
    )
}

pub(crate) fn payload_keys(v: &serde_json::Value) -> Vec<String> {
    let mut keys: Vec<String> = v
        .as_object()
        .expect("payload must be a JSON object")
        .keys()
        .cloned()
        .collect();
    keys.sort();
    keys
}
