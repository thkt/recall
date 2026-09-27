//! Opt-in synthetic measurement, never part of the normal test runtime.
//! Run the command in docs/research/issue-342-scope.md in a release build.

// Keep opt-in measurement helpers inside an explicit test-only item so the
// existing coverage filter excludes the harness, including its ignored test.
#[cfg(test)]
mod tests {
    use std::array;
    use std::cell::RefCell;
    use std::fs;
    use std::path::Path;
    use std::time::{Duration, Instant};

    use rusqlite::{Connection, params};
    use serde_json::json;

    use crate::db::open_db;
    use crate::parser::{self, Source};
    use crate::{ClassifyPhase, read_classification_origin, reclassify_sessions_observed};

    #[derive(Clone, Copy, Debug)]
    enum Mode {
        Legacy,
        SplitFull,
        SplitOrigin,
    }

    #[derive(Default)]
    struct Metrics {
        total: Duration,
        parse: Duration,
        wait: Duration,
        hold: Duration,
        other_wait: Duration,
        other_success: u64,
        retries: u64,
        updates: u64,
    }

    fn full_origin(path: &str, source: &str, id: &str) -> Result<bool, &'static str> {
        let parsed = parser::parse_session_including_empty(
            Path::new(path),
            Source::from_db(source).unwrap(),
        )
        .map_err(|_| "read failure")?
        .ok_or("invalid format")?;
        if parsed.metadata.session_id != id || parsed.origin.identity_conflict {
            return Err("session ID mismatch");
        }
        if !parsed.diagnostics.is_empty() || !parsed.origin.has_records {
            return Err("invalid format");
        }
        Ok(parsed.origin.automated)
    }

    // Baseline is installed from the handoff commit by scripts/measure-classify.py.
    // Refuse direct execution rather than silently measuring a reconstructed control.
    fn legacy(
        _: &mut Connection,
        _: impl FnMut(&str, &str, &str) -> Result<bool, &'static str>,
        _: &RefCell<Metrics>,
    ) {
        panic!("run python3 scripts/measure-classify.py to install the pinned baseline");
    }

    struct Corpus {
        root: tempfile::TempDir,
        sessions: usize,
        records: usize,
        tools: usize,
        bytes: u64,
        file_bytes: Vec<u64>,
    }

    fn corpus(sessions: usize, records: usize, text_bytes: usize, tools: usize) -> Corpus {
        let root = tempfile::tempdir().unwrap();
        let mut bytes = 0;
        let mut file_bytes = Vec::new();
        // Deterministic enumeration (no random generator/seed or private inputs).
        for i in 0..sessions {
            let id = format!("synthetic-{i:04}");
            let automated = i % 4 < 2;
            let source = if i % 2 == 0 {
                Source::Claude
            } else {
                Source::Codex
            };
            let text = format!("Synthetic conversation {i}. {}", "x".repeat(text_bytes));
            let mut lines = String::new();
            if source == Source::Codex {
                lines += &json!({"type":"session_meta","payload":{"id":id,"source":if automated { json!({"internal":"guardian"}) } else { json!("exec") }}}).to_string();
                lines.push('\n');
            }
            for j in 0..records {
                let mut content = vec![json!({"type":"text","text":text})];
                for k in 0..tools {
                    content.push(json!({"type":"tool_use","name":"Edit","input":{"file_path":format!("/synthetic/file-{}", (j * tools + k) % 128),"old_string":"old","new_string":"new"}}));
                }
                let entry = match source {
                    Source::Claude => {
                        json!({"type":"assistant","sessionId":id,"isSidechain":automated,"timestamp":"2026-01-01T00:00:00Z","message":{"content":content}})
                    }
                    Source::Codex => {
                        json!({"type":"response_item","timestamp":"2026-01-01T00:00:00Z","payload":{"role":"assistant","content":content}})
                    }
                };
                lines += &entry.to_string();
                lines.push('\n');
            }
            let size = u64::try_from(lines.len()).unwrap();
            bytes += size;
            file_bytes.push(size);
            fs::write(root.path().join(format!("{id}.jsonl")), lines).unwrap();
        }
        Corpus {
            root,
            sessions,
            records,
            tools,
            bytes,
            file_bytes,
        }
    }

    fn measure(corpus: &Corpus, mode: Mode, repeat: bool, concurrent: bool) -> Metrics {
        // Every trial starts with a fresh, identically seeded product schema (WAL,
        // synchronous=NORMAL); creation/seeding and source generation are untimed.
        let db_dir = tempfile::tempdir().unwrap();
        let db = db_dir.path().join("index.db");
        let mut conn = open_db(&db).unwrap();
        conn.execute_batch(
            "CREATE TABLE classify_probe(value INTEGER); INSERT INTO classify_probe VALUES (0);",
        )
        .unwrap();
        for i in 0..corpus.sessions {
            let id = format!("synthetic-{i:04}");
            let label = repeat.then_some(if i % 4 < 2 {
                "automated"
            } else {
                "interactive"
            });
            conn.execute("INSERT INTO sessions(session_id, source, file_path, session_type) VALUES (?1, ?2, ?3, ?4)", params![id, if i%2==0 {"claude"} else {"codex"}, corpus.root.path().join(format!("{id}.jsonl")).to_str().unwrap(), label]).unwrap();
            conn.execute("INSERT INTO messages(session_id, role, text) VALUES (?1, 'user', 'Synthetic first turn')", [&id]).unwrap();
        }
        let other = open_db(&db).unwrap();
        // A bounded synchronous probe while the log reader is paused. No sleep,
        // scheduler race, or 30-second timeout is included in the parser measurement.
        other.busy_timeout(Duration::ZERO).unwrap();
        let metrics = RefCell::new(Metrics::default());
        let mut probed = false;
        let mut read = |file: Option<fs::File>, path: &str, source: &str, id: &str| {
            let begin = Instant::now();
            let origin = match mode {
                Mode::SplitOrigin => read_classification_origin(file.unwrap(), path, source, id),
                _ => {
                    // Keep the pinned full reader's own open in the controls.
                    drop(file);
                    full_origin(path, source, id)
                }
            };
            metrics.borrow_mut().parse += begin.elapsed();
            if concurrent && !probed {
                probed = true;
                let begin = Instant::now();
                let result = other.execute_batch("BEGIN IMMEDIATE");
                metrics.borrow_mut().other_wait += begin.elapsed();
                match result {
                    Ok(()) => {
                        other
                            .execute("UPDATE classify_probe SET value = value + 1", [])
                            .unwrap();
                        other.execute_batch("COMMIT").unwrap();
                        metrics.borrow_mut().other_success += 1;
                    }
                    Err(error) => assert_eq!(
                        error.sqlite_error_code(),
                        Some(rusqlite::ErrorCode::DatabaseBusy)
                    ),
                }
            }
            origin
        };
        let changes = conn.total_changes();
        let begin = Instant::now();
        if matches!(mode, Mode::Legacy) {
            legacy(
                &mut conn,
                |path, source, id| read(None, path, source, id),
                &metrics,
            );
        } else {
            let mut waiting = None;
            let mut held = None;
            let outcomes = reclassify_sessions_observed(
                &mut conn,
                true,
                false,
                |file, path, source, id| read(Some(file), path, source, id),
                |phase| match phase {
                    ClassifyPhase::WriterWaiting => waiting = Some(Instant::now()),
                    ClassifyPhase::WriterAcquired => {
                        metrics.borrow_mut().wait += waiting.take().unwrap().elapsed();
                        held = Some(Instant::now());
                    }
                    ClassifyPhase::WriterReleased => {
                        metrics.borrow_mut().hold += held.take().unwrap().elapsed()
                    }
                    ClassifyPhase::Retry => metrics.borrow_mut().retries += 1,
                    _ => {}
                },
            )
            .unwrap();
            assert_eq!(outcomes.len(), corpus.sessions);
        }
        let mut metrics = metrics.into_inner();
        metrics.total = begin.elapsed();
        metrics.updates = conn.total_changes() - changes;
        for i in 0..corpus.sessions {
            let label: String = conn
                .query_row(
                    "SELECT session_type FROM sessions WHERE session_id = ?1",
                    [format!("synthetic-{i:04}")],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(
                label,
                if i % 4 < 2 {
                    "automated"
                } else {
                    "interactive"
                }
            );
        }
        assert_eq!(
            metrics.other_success,
            u64::from(concurrent && !matches!(mode, Mode::Legacy))
        );
        assert_eq!(metrics.retries, metrics.other_success);
        assert_eq!(
            metrics.updates,
            if repeat && !matches!(mode, Mode::Legacy) {
                0
            } else {
                u64::try_from(corpus.sessions).unwrap()
            }
        );
        metrics
    }

    fn distribution(samples: &[Metrics], value: impl Fn(&Metrics) -> f64) -> serde_json::Value {
        let mut values: Vec<_> = samples.iter().map(value).collect();
        values.sort_by(f64::total_cmp);
        json!({"median":values[values.len()/2],"min":values[0],"max":values[values.len()-1]})
    }

    #[test]
    #[ignore = "opt-in release performance measurement; uses only generated synthetic logs"]
    fn synthetic_classification_measurement() {
        for (name, sessions, records, text_bytes, tools) in [
            ("few_short", 8, 16, 256, 2),
            ("few_long", 8, 2000, 4096, 8),
            ("many_short", 512, 4, 256, 2),
        ] {
            let corpus = corpus(sessions, records, text_bytes, tools);
            println!(
                "CLASSIFY_CORPUS {}",
                json!({"dataset": name, "sessions":sessions, "records_per_file":records,
            "tools_per_record":tools, "text_padding_bytes":text_bytes, "file_bytes":corpus.file_bytes,
            "sources":"even files Claude; odd files Codex with one extra session_meta record",
            "db_user_rows_per_session":1})
            );
            for repeat in [false, true] {
                for concurrent in [false, true] {
                    let modes = [Mode::Legacy, Mode::SplitFull, Mode::SplitOrigin];
                    // Warm each mode once. Sources are reused; no OS cache eviction.
                    for mode in modes {
                        measure(&corpus, mode, repeat, concurrent);
                    }
                    let mut samples: [Vec<Metrics>; 3] = array::from_fn(|_| Vec::new());
                    // Rotate order across rounds to reduce cache/order bias.
                    for round in 0..5 {
                        for offset in 0..3 {
                            let i = (round + offset) % 3;
                            samples[i].push(measure(&corpus, modes[i], repeat, concurrent));
                        }
                    }
                    for (i, samples) in samples.iter().enumerate() {
                        println!(
                            "CLASSIFY_MEASUREMENT {}",
                            json!({
                                "dataset":name,"sessions":corpus.sessions,"records_per_file":corpus.records,
                                "codex_metadata_records_per_file":1,"tools_per_record":corpus.tools,"total_bytes":corpus.bytes,
                                "mode":format!("{:?}",modes[i]),"repeat":repeat,"concurrent":concurrent,"runs":samples.len(),
                                "sqlite":rusqlite::version(),
                                "total_ms":distribution(samples, |m| m.total.as_secs_f64()*1000.0),
                                "parse_ms":distribution(samples, |m| m.parse.as_secs_f64()*1000.0),
                                "writer_wait_ms":distribution(samples, |m| m.wait.as_secs_f64()*1000.0),
                                "writer_hold_ms":distribution(samples, |m| m.hold.as_secs_f64()*1000.0),
                                "other_writer_wait_ms":distribution(samples, |m| m.other_wait.as_secs_f64()*1000.0),
                                "other_writer_success":distribution(samples, |m| m.other_success as f64),
                                "retries":distribution(samples, |m| m.retries as f64),
                                "physical_updates":distribution(samples, |m| m.updates as f64),
                            })
                        );
                    }
                }
            }
        }
    }
}
