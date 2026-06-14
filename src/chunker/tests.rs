use super::*;

fn msg(role: Role, text: &str) -> Message {
    Message {
        role,
        text: text.to_owned(),
    }
}

/// Pairs each message with a synthetic rowid. `chunk_messages` now takes
/// `&[(i64, Message)]` so the chunker can stamp each chunk's source rowid range
/// (lo = user rowid, hi = adjacent assistant rowid). Callers in production read
/// `SELECT rowid, role, text ... ORDER BY rowid`; these helpers mirror that by
/// numbering messages from `start` in slice order.
fn rowed(messages: Vec<Message>) -> Vec<(i64, Message)> {
    rowed_from(1, messages)
}

fn rowed_from(start: i64, messages: Vec<Message>) -> Vec<(i64, Message)> {
    messages
        .into_iter()
        .enumerate()
        .map(|(i, m)| (start + i as i64, m))
        .collect()
}

#[test]
fn test_004_user_assistant_pair() {
    let messages = rowed(vec![
        msg(Role::User, "認証フローについて教えて"),
        msg(Role::Assistant, "OAuth2を使う方法があります"),
    ]);
    let chunks = chunk_messages("s1", &messages, Some(1000));

    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].content.contains("認証フロー"));
    assert!(chunks[0].content.contains("OAuth2"));
    assert_eq!(chunks[0].session_id, "s1");
    assert_eq!(chunks[0].timestamp, Some(1000));
}

#[test]
fn test_005_user_without_assistant() {
    let messages = rowed(vec![
        msg(Role::User, "最初の質問"),
        msg(Role::User, "次の質問"),
    ]);
    let chunks = chunk_messages("s1", &messages, None);

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].content, "最初の質問");
    assert_eq!(chunks[1].content, "次の質問");
}

#[test]
fn test_006_assistant_only_skipped() {
    let messages = rowed(vec![msg(Role::Assistant, "unsolicited answer")]);
    let chunks = chunk_messages("s1", &messages, None);

    assert!(chunks.is_empty());
}

#[test]
fn test_007_empty_messages() {
    let chunks = chunk_messages("s1", &[], None);
    assert!(chunks.is_empty());
}

#[test]
fn test_mixed_sequence() {
    let messages = rowed(vec![
        msg(Role::User, "Q1"),
        msg(Role::Assistant, "A1"),
        msg(Role::Assistant, "extra A"), // skipped (no preceding user)
        msg(Role::User, "Q2"),           // user-only (next is user)
        msg(Role::User, "Q3"),
        msg(Role::Assistant, "A3"),
    ]);
    let chunks = chunk_messages("s1", &messages, Some(500));

    assert_eq!(chunks.len(), 3);
    assert!(chunks[0].content.contains("Q1"));
    assert!(chunks[0].content.contains("A1"));
    assert_eq!(chunks[1].content, "Q2");
    assert!(chunks[2].content.contains("Q3"));
    assert!(chunks[2].content.contains("A3"));
}

#[test]
fn test_empty_user_text_skipped() {
    let messages = rowed(vec![msg(Role::User, ""), msg(Role::Assistant, "answer")]);
    let chunks = chunk_messages("s1", &messages, None);
    assert!(chunks.is_empty());
}

#[test]
fn test_long_assistant_splits_into_sub_chunks() {
    let long_text = "line\n".repeat(5000); // ~25,000 bytes > MAX_CHUNK_BYTES
    let messages = rowed(vec![
        msg(Role::User, "質問"),
        msg(Role::Assistant, &long_text),
    ]);
    let chunks = chunk_messages("s1", &messages, None);

    assert!(
        chunks.len() > 1,
        "should split into multiple chunks, got {}",
        chunks.len()
    );
    for (i, chunk) in chunks.iter().enumerate() {
        assert!(
            chunk.content.len() <= super::MAX_CHUNK_BYTES,
            "chunk[{i}] is {} bytes, exceeds limit {}",
            chunk.content.len(),
            super::MAX_CHUNK_BYTES
        );
        assert!(
            chunk.content.contains("質問"),
            "chunk[{i}] should contain user question for search context"
        );
    }
    // All assistant text should be preserved across sub-chunks
    let total_content: usize = chunks.iter().map(|c| c.content.len()).sum();
    assert!(total_content > long_text.len(), "no content should be lost");
}

#[test]
fn test_short_content_not_split() {
    let messages = rowed(vec![
        msg(Role::User, "短い質問"),
        msg(Role::Assistant, "短い回答"),
    ]);
    let chunks = chunk_messages("s1", &messages, None);
    assert_eq!(chunks.len(), 1);
}

#[test]
fn test_user_near_limit_multibyte_assistant_terminates() {
    // Regression: user_text within 1-3 bytes of MAX_CHUNK_BYTES leaves `available` < 4.
    // When assistant text leads with a multibyte char wider than `available`,
    // find_split_boundary returns 0, so the sub-chunk loop never advances —
    // an infinite loop that also pushes a ~16KB header every iteration (OOM).
    let user = "a".repeat(15998); // header = 15999 bytes → available = 1
    let assistant = "こんにちは世界"; // 3-byte CJK lead char > available
    let messages = rowed(vec![
        msg(Role::User, &user),
        msg(Role::Assistant, assistant),
    ]);
    let chunks = chunk_messages("s1", &messages, None);
    assert!(!chunks.is_empty(), "must terminate and produce chunks");
    assert!(
        chunks.iter().any(|c| c.content.contains("こんにちは")),
        "assistant content must be preserved across the split"
    );
    for (i, chunk) in chunks.iter().enumerate() {
        assert!(
            chunk.content.len() <= MAX_CHUNK_BYTES,
            "chunk[{i}] is {} bytes, exceeds limit {MAX_CHUNK_BYTES}",
            chunk.content.len()
        );
    }
}

#[test]
fn test_available_4_takes_sub_chunk_path() {
    // Boundary: available == 4 is the smallest value that enters the sub-chunk
    // branch (>= 4), where each chunk carries user_text as a prefix. One byte less
    // (available == 3) falls to the independent-split branch instead.
    let user = "a".repeat(15995); // header = 15996 bytes → available = 4
    let assistant = "日本語のテキスト"; // exceeds available → forces a split
    let messages = rowed(vec![
        msg(Role::User, &user),
        msg(Role::Assistant, assistant),
    ]);
    let chunks = chunk_messages("s1", &messages, None);

    assert!(chunks.len() > 1, "must split, got {}", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        assert!(
            chunk.content.starts_with(&user),
            "chunk[{i}] must carry user_text as prefix (sub-chunk path)"
        );
        assert!(
            chunk.content.len() <= MAX_CHUNK_BYTES,
            "chunk[{i}] is {} bytes, exceeds limit {MAX_CHUNK_BYTES}",
            chunk.content.len()
        );
    }
}

// FR-1/FR-2 (#192), acceptance "chunker の lo/hi 付与 (user-only) を chunker
// 単体テストで pin". A user message with no adjacent assistant gets a chunk
// whose source rowid range collapses to a point: lo == hi == the user message's
// own rowid. Perspective: Boundary (the degenerate single-row range).
#[test]
fn test_192_chunk_user_only_lo_eq_hi_eq_user_rowid() {
    // rowid 7 user, rowid 8 user → first is user-only (next message is a user).
    let messages = rowed_from(
        7,
        vec![msg(Role::User, "standalone"), msg(Role::User, "next")],
    );
    let chunks = chunk_messages("s1", &messages, None);

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].content, "standalone");
    assert_eq!(
        chunks[0].src_rowid_lo, 7,
        "user-only chunk lo is the user message rowid"
    );
    assert_eq!(
        chunks[0].src_rowid_hi, 7,
        "user-only chunk hi collapses to the user message rowid (no assistant)"
    );
    // The second user-only chunk takes its own rowid.
    assert_eq!(chunks[1].src_rowid_lo, 8);
    assert_eq!(chunks[1].src_rowid_hi, 8);
}

// FR-1/FR-2 (#192), acceptance "chunker の lo/hi 付与 (user+assistant)". A user
// message followed by an adjacent assistant forms one chunk spanning both rowids:
// lo = user rowid, hi = assistant rowid. Perspective: Equivalence (the common
// Q&A pair) + Boundary (lo < hi span).
#[test]
fn test_192_chunk_user_assistant_spans_user_lo_to_assistant_hi() {
    // rowid 3 user, rowid 4 assistant → one chunk with range [3, 4].
    let messages = rowed_from(
        3,
        vec![
            msg(Role::User, "how do I rotate tokens"),
            msg(Role::Assistant, "use a refresh token grant"),
        ],
    );
    let chunks = chunk_messages("s1", &messages, None);

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].src_rowid_lo, 3, "lo is the user message rowid");
    assert_eq!(
        chunks[0].src_rowid_hi, 4,
        "hi is the adjacent assistant message rowid"
    );
}

// FR-2 (#192), acceptance "chunker の lo/hi 付与 (split)". When one user+assistant
// pair is split into N sub-chunks (oversized assistant), every sibling chunk
// shares the same [lo, hi] range — the whole group originates from the same two
// message rowids. Perspective: Combination (split path × rowid stamping) +
// Hazard (a per-sub-chunk rowid would mis-scope the search JOIN).
#[test]
fn test_192_split_chunks_share_one_lo_hi_range() {
    let long_text = "line\n".repeat(5000); // forces a multi-sub-chunk split
    // rowid 10 user, rowid 11 assistant.
    let messages = rowed_from(
        10,
        vec![msg(Role::User, "質問"), msg(Role::Assistant, &long_text)],
    );
    let chunks = chunk_messages("s1", &messages, None);

    assert!(
        chunks.len() > 1,
        "precondition: oversized assistant must split, got {}",
        chunks.len()
    );
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(
            chunk.src_rowid_lo, 10,
            "sub-chunk[{i}] shares the group's user rowid as lo"
        );
        assert_eq!(
            chunk.src_rowid_hi, 11,
            "sub-chunk[{i}] shares the group's assistant rowid as hi"
        );
    }
}
