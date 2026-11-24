#![cfg(test)]
use std::assert_matches;
use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use super::*;
use crate::envelope::Note;

struct FailingWriter;

impl Write for FailingWriter {
    fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("stdout unavailable"))
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn emit_success_propagates_non_pipe_write_error() {
    let out = CommandOutput::ok("hello".to_owned(), serde_json::Value::Null);
    let mut writer = FailingWriter;

    let err = emit_success_to(&out, false, &mut writer, &mut io::sink()).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Other);
}

struct ClosedPipe;

impl Write for ClosedPipe {
    fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
        Err(io::Error::from(ErrorKind::BrokenPipe))
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn result_with_notes() -> CommandOutput {
    CommandOutput::with_notes(
        "result\n".to_owned(),
        serde_json::json!({}),
        true,
        vec![Note::shown("already printed"), Note::new("stale snapshot")],
    )
}

#[test]
fn a_text_run_writes_its_result_then_the_notes_it_has_not_shown() {
    let (mut stdout, mut stderr) = (Vec::new(), Vec::new());

    let written = emit_success_to(&result_with_notes(), false, &mut stdout, &mut stderr).unwrap();

    assert_eq!(
        (
            written,
            String::from_utf8(stdout).unwrap(),
            String::from_utf8(stderr).unwrap()
        ),
        (
            WriteOutcome::Written,
            "result\n".to_owned(),
            "note: stale snapshot\n".to_owned()
        )
    );
}

#[test]
fn a_json_run_leaves_its_notes_in_the_envelope() {
    let mut stderr = Vec::new();

    emit_success_to(&result_with_notes(), true, &mut io::sink(), &mut stderr).unwrap();

    assert_eq!(stderr, b"");
}

#[test]
fn a_text_run_whose_stderr_is_closed_still_reports_its_result_written() {
    let written = emit_success_to(
        &result_with_notes(),
        false,
        &mut Vec::new(),
        &mut ClosedPipe,
    )
    .unwrap();

    assert_eq!(written, WriteOutcome::Written);
}

#[test]
fn a_text_run_whose_stdout_is_closed_still_prints_its_notes() {
    let mut stderr = Vec::new();

    let written =
        emit_success_to(&result_with_notes(), false, &mut ClosedPipe, &mut stderr).unwrap();

    assert_eq!(
        (written, String::from_utf8(stderr).unwrap()),
        (
            WriteOutcome::PipeClosed,
            "note: stale snapshot\n".to_owned()
        )
    );
}

#[test]
fn a_text_run_whose_result_fails_to_write_prints_no_notes() {
    let mut stderr = Vec::new();

    emit_success_to(&result_with_notes(), false, &mut FailingWriter, &mut stderr).unwrap_err();

    assert_eq!(stderr, b"");
}

#[test]
fn exit_code_for_write_maps_write_failure_to_io_error() {
    let code = exit_code_for_write(Err(io::Error::other("stdout unavailable")), false);
    assert_eq!(code, ExitCode::from(ErrorCode::IoError.code()));
}

#[test]
fn exit_code_for_write_maps_written_to_success() {
    let code = exit_code_for_write(Ok(WriteOutcome::Written), false);
    assert_eq!(code, ExitCode::SUCCESS);
}

#[test]
fn search_without_a_query_or_file_is_rejected_when_arguments_are_parsed() {
    assert!(Cli::try_parse_from(["recall", "search"]).is_err());
    assert!(Cli::try_parse_from(["recall", "search", "--file", "a.rs"]).is_ok());
    assert!(Cli::try_parse_from(["recall", "search", "auth"]).is_ok());
}

#[test]
fn rebuild_parses_as_its_own_subcommand() {
    let cli = Cli::try_parse_from(["recall", "rebuild"]).unwrap();
    assert_matches!(cli.command, Some(Command::Rebuild { .. }));
}

#[test]
fn rebuild_is_not_rewritten_into_a_search_query() {
    let args: Vec<OsString> = ["recall", "rebuild"].iter().map(OsString::from).collect();
    assert!(try_expand_shorthand(&args).is_none());
}

#[test]
fn doctor_recognized_by_shorthand_expansion() {
    let args: Vec<OsString> = ["recall", "doctor"].iter().map(OsString::from).collect();
    assert!(try_expand_shorthand(&args).is_none());
    let cli = Cli::try_parse_from(["recall", "doctor"]).unwrap();
    assert_matches!(cli.command, Some(Command::Doctor { fix: false }));
}

#[test]
fn doctor_fix_flag_parses() {
    let cli = Cli::try_parse_from(["recall", "doctor", "--fix"]).unwrap();
    assert_matches!(cli.command, Some(Command::Doctor { fix: true }));
}

#[test]
fn embed_token_shorthand_expands_to_search() {
    let args: Vec<OsString> = ["recall", "embed"].iter().map(OsString::from).collect();
    let expected: Vec<OsString> = ["recall", "search", "embed"]
        .iter()
        .map(OsString::from)
        .collect();
    assert_eq!(try_expand_shorthand(&args), Some(expected));
}

#[test]
fn known_subcommands_covers_every_clap_subcommand() {
    use clap::CommandFactory;
    for sub in Cli::command().get_subcommands() {
        let name = sub.get_name();
        assert!(
            KNOWN_SUBCOMMANDS.contains(&name),
            "clap subcommand `{name}` is missing from KNOWN_SUBCOMMANDS; \
                 shorthand expansion will rewrite `recall {name}` to a search query"
        );
    }
}

#[test]
fn every_command_a_message_tells_the_user_to_run_exists() {
    let mut unknown = Vec::new();
    for (file, word) in commands_named_in(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"))) {
        if !KNOWN_SUBCOMMANDS.contains(&word.as_str()) {
            unknown.push(format!("{}: `recall {word}`", file.display()));
        }
    }
    assert!(
        unknown.is_empty(),
        "messages name commands that do not exist, which shorthand expansion runs as a search query (`recall embed` becomes `recall search embed`): {unknown:?}"
    );
}

fn commands_named_in(dir: &Path) -> Vec<(PathBuf, String)> {
    let mut named = Vec::new();
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if path.is_dir() {
            named.extend(commands_named_in(&path));
        } else if name.ends_with(".rs") && !name.ends_with("tests.rs") {
            let source = fs::read_to_string(&path).unwrap();
            for rest in source.split("`recall ").skip(1) {
                let word: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_lowercase() || *c == '-')
                    .collect();
                if !word.is_empty() && !word.starts_with('-') {
                    named.push((path.clone(), word));
                }
            }
        }
    }
    named
}

#[test]
fn search_result_and_session_data_stay_non_serialize() {
    fn leading_block_mentions_serialize(src: &str, decl: &str) -> bool {
        let pos = src.find(decl).unwrap_or_else(|| {
            panic!("{decl} not found — the Serialize guard is reading the wrong source")
        });
        let block_start = src[..pos].rfind(['}', ';']).map(|i| i + 1).unwrap_or(0);
        src[block_start..pos].contains("Serialize")
    }

    assert!(
        !leading_block_mentions_serialize(include_str!("../search.rs"), "pub struct SearchResult"),
        "SearchResult must not derive Serialize (ADR-0001): the --json schema is built explicitly"
    );
    assert!(
        !leading_block_mentions_serialize(
            include_str!("../parser/mod.rs"),
            "pub struct SessionData"
        ),
        "SessionData must not derive Serialize (ADR-0001): the --json schema is built explicitly"
    );
}

fn output_with_text(text: &str) -> CommandOutput {
    CommandOutput::with_notes(
        text.to_owned(),
        serde_json::json!({"messages": [{"text": "回答\n\"引用\""}]}),
        true,
        vec![Note::new("diagnostic\nwith \"quotes\"")],
    )
}

#[test]
fn text_output_ends_in_exactly_the_newlines_it_had_and_adds_one_only_when_missing() {
    for (text, printed) in [
        ("", ""),
        ("質問と回答", "質問と回答\n"),
        ("answer\n", "answer\n"),
        ("answer\n\n", "answer\n\n"),
    ] {
        let mut buf = Vec::new();
        assert_eq!(
            emit_success_to(&output_with_text(text), false, &mut buf, &mut io::sink()).unwrap(),
            WriteOutcome::Written,
        );
        assert_eq!(String::from_utf8(buf).unwrap(), printed, "text: {text:?}");
    }
}

#[test]
fn json_output_prints_one_newline_terminated_envelope_with_data_degraded_and_notes() {
    let mut buf = Vec::new();
    assert_eq!(
        emit_success_to(
            &output_with_text("質問と回答"),
            true,
            &mut buf,
            &mut io::sink()
        )
        .unwrap(),
        WriteOutcome::Written,
    );
    let printed = String::from_utf8(buf).unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&printed).unwrap(),
        serde_json::json!({
            "data": {"messages": [{"text": "回答\n\"引用\""}]},
            "degraded": true,
            "notes": ["diagnostic\nwith \"quotes\""],
        }),
    );
    assert!(printed.ends_with("}\n"), "{printed:?}");
}
