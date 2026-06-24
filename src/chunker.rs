use crate::parser::{Message, Role};

pub(crate) struct QAChunk {
    pub session_id: String,
    /// Embedding content: user_text [+ "\n" + assistant_text]. Prefix added by embedder.
    pub content: String,
    pub timestamp: Option<i64>,
    /// Source message rowid range (#192): lo = user message rowid, hi = last
    /// consecutive assistant rowid (or lo when user-only). fetch_chunks resolves an
    /// FTS hit to its owning chunk via `m.rowid BETWEEN lo AND hi`; this replaces
    /// the instr substring guess for linked chunks, while legacy NULL-range rows
    /// still fall back to instr. Split-siblings of one message share one range.
    pub src_rowid_lo: i64,
    pub src_rowid_hi: i64,
}

/// Max content bytes per chunk (~4000-5000 tokens, well under model's 8192 limit).
/// Prevents O(seq_len²) attention explosion from oversized QA pairs.
const MAX_CHUNK_BYTES: usize = 16_000;

fn split_text(text: &str, max_bytes: usize) -> Vec<String> {
    if text.len() <= max_bytes {
        return vec![text.to_owned()];
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
            // Need room for at least one full char (max 4 UTF-8 bytes); below that,
            // find_split_boundary returns 0 and the sub-chunk loop cannot advance.
            if available >= 4 {
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
            // user_text leaves < 4 bytes of room (or exceeds the limit);
            // split user and assistant independently.
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

/// Build Q&A chunks from a session's messages.
///
/// Rules:
/// - user + following assistant run → one chunk; consecutive assistants merge
/// - user followed by user (no assistant) → user-only chunk
/// - assistant without preceding user → skipped
/// - empty messages → empty list
pub(crate) fn chunk_messages(
    session_id: &str,
    messages: &[(i64, Message)],
    timestamp: Option<i64>,
) -> Vec<QAChunk> {
    let mut chunks = Vec::new();
    let mut i = 0;

    while i < messages.len() {
        let (lo, msg) = (messages[i].0, &messages[i].1);

        if msg.role != Role::User {
            // Skip assistant without preceding user
            i += 1;
            continue;
        }

        if msg.text.is_empty() {
            i += 1;
            continue;
        }

        // hi defaults to the user rowid (user-only group), extended to the last
        // consumed assistant rowid below. Consecutive assistants arise when the
        // parser drops tool_result-only user turns between them (#229); merge
        // them all into this Q&A so the trailing answer is not orphaned out of
        // the embedding path.
        let mut hi = lo;
        let mut assistant_parts = Vec::new();
        while i + 1 < messages.len() && messages[i + 1].1.role == Role::Assistant {
            i += 1;
            hi = messages[i].0;
            let text = messages[i].1.text.as_str();
            if !text.is_empty() {
                assistant_parts.push(text);
            }
        }
        let merged_assistant = (!assistant_parts.is_empty()).then(|| assistant_parts.join("\n"));

        let contents = build_content(&msg.text, merged_assistant.as_deref());
        for content in contents {
            chunks.push(QAChunk {
                session_id: session_id.to_owned(),
                content,
                timestamp,
                src_rowid_lo: lo,
                src_rowid_hi: hi,
            });
        }

        i += 1;
    }

    chunks
}

#[cfg(test)]
mod tests;
