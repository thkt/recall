use super::*;

fn user(text: &str) -> Message {
    Message {
        role: Role::User,
        text: text.to_owned(),
    }
}

fn assistant(text: &str) -> Message {
    Message {
        role: Role::Assistant,
        text: text.to_owned(),
    }
}

/// A chunk of session `s1` with no timestamp, covering message rows `lo..=hi`.
fn chunk(content: &str, lo: i64, hi: i64) -> QAChunk {
    QAChunk {
        session_id: "s1".to_owned(),
        content: content.to_owned(),
        timestamp: None,
        src_rowid_lo: lo,
        src_rowid_hi: hi,
    }
}

#[test]
fn a_question_and_its_answer_become_one_chunk_spanning_both_rows() {
    let messages = [
        (3, user("how do I rotate tokens")),
        (4, assistant("use a refresh token grant")),
    ];

    assert_eq!(
        chunk_messages("s1", &messages, Some(1000)),
        [QAChunk {
            session_id: "s1".to_owned(),
            content: "how do I rotate tokens\nuse a refresh token grant".to_owned(),
            timestamp: Some(1000),
            src_rowid_lo: 3,
            src_rowid_hi: 4,
        }]
    );
}

#[test]
fn a_question_without_an_answer_becomes_a_chunk_of_its_own() {
    let messages = [(7, user("standalone")), (8, user("next"))];

    assert_eq!(
        chunk_messages("s1", &messages, None),
        [chunk("standalone", 7, 7), chunk("next", 8, 8)]
    );
}

#[test]
fn answers_without_a_question_and_empty_questions_are_not_chunked() {
    assert_eq!(chunk_messages("s1", &[], None), []);
    assert_eq!(
        chunk_messages("s1", &[(1, assistant("unsolicited answer"))], None),
        []
    );
    assert_eq!(
        chunk_messages("s1", &[(1, user("")), (2, assistant("answer"))], None),
        []
    );
}

#[test]
fn consecutive_answers_merge_into_the_chunk_of_the_question_before_them() {
    let messages = [
        (1, user("Q1")),
        (2, assistant("A1")),
        (3, assistant("extra A")),
        (4, user("Q2")),
        (5, user("Q3")),
        (6, assistant("A3")),
    ];

    assert_eq!(
        chunk_messages("s1", &messages, None),
        [
            chunk("Q1\nA1\nextra A", 1, 3),
            chunk("Q2", 4, 4),
            chunk("Q3\nA3", 5, 6),
        ]
    );
}

#[test]
fn empty_answers_add_no_text_but_still_extend_the_chunk_range() {
    let trailing = [
        (1, user("Q")),
        (2, assistant("real answer")),
        (3, assistant("")),
    ];
    let leading = [
        (1, user("Q")),
        (2, assistant("")),
        (3, assistant("real answer")),
    ];
    let only_empty = [(1, user("Q")), (2, assistant("")), (3, assistant(""))];

    assert_eq!(
        chunk_messages("s1", &trailing, None),
        [chunk("Q\nreal answer", 1, 3)]
    );
    assert_eq!(
        chunk_messages("s1", &leading, None),
        [chunk("Q\nreal answer", 1, 3)]
    );
    assert_eq!(chunk_messages("s1", &only_empty, None), [chunk("Q", 1, 3)]);
}

#[test]
fn an_oversized_answer_splits_at_line_breaks_and_each_part_repeats_the_question() {
    let answer = "line\n".repeat(5000);
    let messages = [(10, user("質問")), (11, assistant(&answer))];

    assert_eq!(
        chunk_messages("s1", &messages, None),
        [
            chunk(&format!("質問\n{}", "line\n".repeat(3198)), 10, 11),
            chunk(&format!("質問\n{}", "line\n".repeat(1802)), 10, 11),
        ]
    );
}

#[test]
fn an_oversized_merged_answer_splits_at_the_paragraph_break_between_answers() {
    let half = "line\n".repeat(2000);
    let messages = [
        (1, user("質問")),
        (2, assistant(&half)),
        (3, assistant(&half)),
    ];

    assert_eq!(
        chunk_messages("s1", &messages, None),
        [
            chunk(&format!("質問\n{half}\n"), 1, 3),
            chunk(&format!("質問\n{half}"), 1, 3),
        ]
    );
}

#[test]
fn a_split_keeps_every_character_with_and_without_the_question_prefix() {
    let first = "界".repeat(4000);
    let second = "末".repeat(2000);
    let text = format!("{first}\n\n{second}");

    assert_eq!(
        chunk_messages("s1", &[(1, user(&text))], None),
        [chunk(&format!("{first}\n\n"), 1, 1), chunk(&second, 1, 1)]
    );
    assert_eq!(
        chunk_messages("s1", &[(1, user("質問")), (2, assistant(&text))], None),
        [
            chunk(&format!("質問\n{first}\n\n"), 1, 2),
            chunk(&format!("質問\n{second}"), 1, 2),
        ]
    );
}

#[test]
fn a_question_too_long_to_repeat_is_chunked_apart_from_its_answer() {
    let question = "a".repeat(15_998);
    let messages = [(1, user(&question)), (2, assistant("こんにちは世界"))];

    assert_eq!(
        chunk_messages("s1", &messages, None),
        [chunk(&question, 1, 2), chunk("こんにちは世界", 1, 2)]
    );
}

#[test]
fn a_question_that_leaves_room_for_one_character_repeats_before_each_character() {
    let question = "a".repeat(15_995);
    let messages = [(1, user(&question)), (2, assistant("日本語のテキスト"))];

    assert_eq!(
        chunk_messages("s1", &messages, None),
        [
            chunk(&format!("{question}\n日"), 1, 2),
            chunk(&format!("{question}\n本"), 1, 2),
            chunk(&format!("{question}\n語"), 1, 2),
            chunk(&format!("{question}\nの"), 1, 2),
            chunk(&format!("{question}\nテ"), 1, 2),
            chunk(&format!("{question}\nキ"), 1, 2),
            chunk(&format!("{question}\nス"), 1, 2),
            chunk(&format!("{question}\nト"), 1, 2),
        ]
    );
}
