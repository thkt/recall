use sha2::{Digest, Sha256};

use crate::parser::{Message, Role};

pub(crate) struct QAChunk {
    pub session_id: String,
    pub user_text: String,
    pub assistant_text: Option<String>,
    /// Embedding content: user_text [+ "\n" + assistant_text]. Prefix added by embedder.
    pub content: String,
    pub timestamp: Option<i64>,
    pub chunk_hash: String,
}

/// Max content bytes per chunk (~4000-5000 tokens, well under model's 8192 limit).
/// Prevents O(seq_len²) attention explosion from oversized QA pairs.
const MAX_CHUNK_BYTES: usize = 16_000;

fn split_text(text: &str, max_bytes: usize) -> Vec<String> {
    if text.len() <= max_bytes {
        return vec![text.to_string()];
    }
    let mut chunks = Vec::new();
    let mut remaining = text;
    while !remaining.is_empty() {
        let split_at = find_split_boundary(remaining, max_bytes);
        chunks.push(remaining[..split_at].to_string());
        remaining = &remaining[split_at..];
        remaining = remaining.trim_start_matches('\n');
    }
    chunks
}

fn build_content(user_text: &str, assistant_text: Option<&str>) -> Vec<String> {
    match assistant_text {
        None => split_text(user_text, MAX_CHUNK_BYTES),
        Some(a) => {
            let header = format!("{user_text}\n");
            if header.len() + a.len() <= MAX_CHUNK_BYTES {
                return vec![format!("{header}{a}")];
            }
            // Split assistant text with user_text as prefix per sub-chunk
            let available = MAX_CHUNK_BYTES.saturating_sub(header.len());
            if available > 0 {
                let mut chunks = Vec::new();
                let mut remaining = a;
                while !remaining.is_empty() {
                    let split_at = find_split_boundary(remaining, available);
                    chunks.push(format!("{header}{}", &remaining[..split_at]));
                    remaining = &remaining[split_at..];
                    remaining = remaining.trim_start_matches('\n');
                }
                return chunks;
            }
            // user_text alone exceeds limit; split both independently
            let mut chunks = split_text(user_text, MAX_CHUNK_BYTES);
            chunks.extend(split_text(a, MAX_CHUNK_BYTES));
            chunks
        }
    }
}

/// Find a byte position to split text, preferring line boundaries.
fn find_split_boundary(text: &str, max_bytes: usize) -> usize {
    if text.len() <= max_bytes {
        return text.len();
    }
    let end = text.floor_char_boundary(max_bytes);
    // Prefer paragraph boundary, then line boundary
    if let Some(pos) = text[..end].rfind("\n\n") {
        return pos + 2;
    }
    match text[..end].rfind('\n') {
        Some(pos) => pos + 1,
        None => end,
    }
}

fn sha256_hex(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Build Q&A chunks from a session's messages.
///
/// Rules:
/// - user + adjacent assistant → one chunk with both texts
/// - user followed by user (no assistant) → user-only chunk
/// - assistant without preceding user → skipped
/// - empty messages → empty list
pub(crate) fn chunk_messages(
    session_id: &str,
    messages: &[Message],
    timestamp: Option<i64>,
) -> Vec<QAChunk> {
    let mut chunks = Vec::new();
    let mut i = 0;

    while i < messages.len() {
        let msg = &messages[i];

        if msg.role != Role::User {
            // Skip assistant without preceding user
            i += 1;
            continue;
        }

        if msg.text.is_empty() {
            i += 1;
            continue;
        }

        let assistant_text = if i + 1 < messages.len() && messages[i + 1].role == Role::Assistant {
            i += 1;
            let text = messages[i].text.as_str();
            if text.is_empty() { None } else { Some(text) }
        } else {
            None
        };

        let contents = build_content(&msg.text, assistant_text);
        for content in contents {
            let chunk_hash = sha256_hex(&content);
            chunks.push(QAChunk {
                session_id: session_id.to_string(),
                user_text: msg.text.clone(),
                assistant_text: assistant_text.map(String::from),
                content,
                timestamp,
                chunk_hash,
            });
        }

        i += 1;
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: Role, text: &str) -> Message {
        Message {
            role,
            text: text.to_string(),
        }
    }

    #[test]
    fn test_004_user_assistant_pair() {
        let messages = vec![
            msg(Role::User, "認証フローについて教えて"),
            msg(Role::Assistant, "OAuth2を使う方法があります"),
        ];
        let chunks = chunk_messages("s1", &messages, Some(1000));

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].user_text, "認証フローについて教えて");
        assert_eq!(
            chunks[0].assistant_text.as_deref(),
            Some("OAuth2を使う方法があります")
        );
        assert!(chunks[0].content.contains("認証フロー"));
        assert!(chunks[0].content.contains("OAuth2"));
        assert_eq!(chunks[0].session_id, "s1");
        assert_eq!(chunks[0].timestamp, Some(1000));
        assert!(!chunks[0].chunk_hash.is_empty());
    }

    #[test]
    fn test_005_user_without_assistant() {
        let messages = vec![msg(Role::User, "最初の質問"), msg(Role::User, "次の質問")];
        let chunks = chunk_messages("s1", &messages, None);

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].user_text, "最初の質問");
        assert!(chunks[0].assistant_text.is_none());
        assert_eq!(chunks[1].user_text, "次の質問");
    }

    #[test]
    fn test_006_assistant_only_skipped() {
        let messages = vec![msg(Role::Assistant, "unsolicited answer")];
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
        let messages = vec![
            msg(Role::User, "Q1"),
            msg(Role::Assistant, "A1"),
            msg(Role::Assistant, "extra A"), // skipped (no preceding user)
            msg(Role::User, "Q2"),           // user-only (next is user)
            msg(Role::User, "Q3"),
            msg(Role::Assistant, "A3"),
        ];
        let chunks = chunk_messages("s1", &messages, Some(500));

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].user_text, "Q1");
        assert_eq!(chunks[0].assistant_text.as_deref(), Some("A1"));
        assert_eq!(chunks[1].user_text, "Q2");
        assert!(chunks[1].assistant_text.is_none());
        assert_eq!(chunks[2].user_text, "Q3");
        assert_eq!(chunks[2].assistant_text.as_deref(), Some("A3"));
    }

    #[test]
    fn test_empty_user_text_skipped() {
        let messages = vec![msg(Role::User, ""), msg(Role::Assistant, "answer")];
        let chunks = chunk_messages("s1", &messages, None);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_hash_deterministic() {
        let messages = vec![msg(Role::User, "same question")];
        let c1 = chunk_messages("s1", &messages, None);
        let c2 = chunk_messages("s2", &messages, None);
        assert_eq!(
            c1[0].chunk_hash, c2[0].chunk_hash,
            "same content → same hash"
        );
    }

    #[test]
    fn test_long_assistant_splits_into_sub_chunks() {
        let long_text = "line\n".repeat(5000); // ~25,000 bytes > MAX_CHUNK_BYTES
        let messages = vec![msg(Role::User, "質問"), msg(Role::Assistant, &long_text)];
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
        let messages = vec![
            msg(Role::User, "短い質問"),
            msg(Role::Assistant, "短い回答"),
        ];
        let chunks = chunk_messages("s1", &messages, None);
        assert_eq!(chunks.len(), 1);
    }
}
