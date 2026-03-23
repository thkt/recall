use sha2::{Digest, Sha256};

use crate::embedder::DOCUMENT_PREFIX;
use crate::parser::{Message, Role};

pub(crate) struct QAChunk {
    pub session_id: String,
    pub user_text: String,
    pub assistant_text: Option<String>,
    /// "検索文書: " + user_text + "\n" + assistant_text
    pub content: String,
    pub timestamp: Option<i64>,
    pub chunk_hash: String,
}

fn build_content(user_text: &str, assistant_text: Option<&str>) -> String {
    match assistant_text {
        Some(a) => format!("{DOCUMENT_PREFIX}{user_text}\n{a}"),
        None => format!("{DOCUMENT_PREFIX}{user_text}"),
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
            Some(messages[i].text.as_str())
        } else {
            None
        };

        let content = build_content(&msg.text, assistant_text);
        let chunk_hash = sha256_hex(&content);

        chunks.push(QAChunk {
            session_id: session_id.to_string(),
            user_text: msg.text.clone(),
            assistant_text: assistant_text.map(String::from),
            content,
            timestamp,
            chunk_hash,
        });

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

    // T-004: user + assistant pair → QAChunk with both texts (FR-003)
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
        assert!(chunks[0].content.starts_with(DOCUMENT_PREFIX));
        assert!(chunks[0].content.contains("認証フロー"));
        assert!(chunks[0].content.contains("OAuth2"));
        assert_eq!(chunks[0].session_id, "s1");
        assert_eq!(chunks[0].timestamp, Some(1000));
        assert!(!chunks[0].chunk_hash.is_empty());
    }

    // T-005: user without assistant → user-only chunk (FR-003)
    #[test]
    fn test_005_user_without_assistant() {
        let messages = vec![msg(Role::User, "最初の質問"), msg(Role::User, "次の質問")];
        let chunks = chunk_messages("s1", &messages, None);

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].user_text, "最初の質問");
        assert!(chunks[0].assistant_text.is_none());
        assert_eq!(chunks[1].user_text, "次の質問");
    }

    // T-006: assistant only → skip (FR-003)
    #[test]
    fn test_006_assistant_only_skipped() {
        let messages = vec![msg(Role::Assistant, "unsolicited answer")];
        let chunks = chunk_messages("s1", &messages, None);

        assert!(chunks.is_empty());
    }

    // T-007: empty messages → empty list (FR-003)
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
}
