use std::fs;

use amici::testing::hybrid::assert_filter_symmetric;

use super::*;
use crate::db::setup_test_db;
use crate::embedder::MockEmbedder;
use crate::indexer::index_chunks;

fn make_result(sid: &str, ts: i64) -> SearchResult {
    SearchResult {
        session: SessionData {
            session_id: sid.to_owned(),
            source: Source::Claude,
            file_path: String::new(),
            project: String::new(),
            slug: String::new(),
            timestamp: Some(ts),
        },
        excerpt: String::new(),
    }
}

fn insert_session(conn: &Connection, sid: &str, source: &str, project: &str, ts: i64) {
    conn.execute(
            "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) VALUES (?1, ?2, '', ?3, ?4, ?5, 0.0)",
            rusqlite::params![sid, source, project, sid, ts],
        ).unwrap();
}

fn insert_message(conn: &Connection, sid: &str, role: &str, text: &str) {
    conn.execute(
        "INSERT INTO messages (session_id, role, text) VALUES (?1, ?2, ?3)",
        rusqlite::params![sid, role, text],
    )
    .unwrap();
}

fn insert_chunk_with_embedding(conn: &Connection, chunk_id: i64, sid: &str, text: &str, ts: i64) {
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp) VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![chunk_id, sid, text, ts],
    )
    .unwrap();
    let embedding = MockEmbedder::deterministic_vector(text);
    let embedding_bytes: &[u8] = f32_as_bytes(&embedding);
    conn.execute(
        "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?1, ?2)",
        rusqlite::params![chunk_id, embedding_bytes],
    )
    .unwrap();
}

/// Like `insert_chunk_with_embedding`, but stamps the chunk's source message
/// rowid range (#192). The primary `fetch_chunks` SQL joins the FTS top-hit
/// `m.rowid BETWEEN c.src_rowid_lo AND c.src_rowid_hi`, so the [lo, hi] literals
/// must bracket the intended message's rowid. Messages are inserted by
/// `insert_message` in order, taking rowids 1, 2, 3, … — assert with
/// `message_rowid` when the alignment is load-bearing.
fn insert_chunk_with_rowid_range(
    conn: &Connection,
    chunk_id: i64,
    sid: &str,
    text: &str,
    ts: i64,
    lo: i64,
    hi: i64,
) {
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp, src_rowid_lo, src_rowid_hi) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![chunk_id, sid, text, ts, lo, hi],
    )
    .unwrap();
    let embedding = MockEmbedder::deterministic_vector(text);
    let embedding_bytes: &[u8] = f32_as_bytes(&embedding);
    conn.execute(
        "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?1, ?2)",
        rusqlite::params![chunk_id, embedding_bytes],
    )
    .unwrap();
}

/// The rowid FTS5 assigned to a message, so a test can pin lo/hi alignment to the
/// actual stored rowid rather than trusting insertion order.
fn message_rowid(conn: &Connection, sid: &str, text: &str) -> i64 {
    conn.query_row(
        "SELECT rowid FROM messages WHERE session_id = ?1 AND text = ?2",
        rusqlite::params![sid, text],
        |r| r.get(0),
    )
    .unwrap()
}

#[test]
fn test_basic_search_returns_match() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "how to implement authentication");

    let results = search(&conn, "authentication", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session.session_id, "s1");
    assert!(!results[0].excerpt.is_empty());
}

// T-001 (#8): an FTS-hit excerpt is the full qa_chunk content, not a 20-token
// snippet of the matched message. The chunk holds the matched message text
// plus the rest of the Q&A; fetch_chunks locates it via instr(content,
// message_text) and returns the whole chunk.
#[test]
fn test_fts_hit_excerpt_is_full_chunk_not_snippet() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "how to implement authentication");
    insert_chunk_with_embedding(
        &conn,
        1,
        "s1",
        "how to implement authentication — use JWT with short-lived access tokens and rotating refresh tokens",
        1709251200000,
    );

    let results = search(&conn, "authentication", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    let excerpt = &results[0].excerpt;
    assert!(
        excerpt.contains("rotating refresh tokens"),
        "FTS-hit excerpt must be the full chunk, not a 20-token snippet: {excerpt:?}"
    );
}

// T-002 (#8): a multi-term (implicit-AND) query still resolves to the
// containing chunk. MATCH applies the operator; instr only sees the concrete
// matched message text, so the operator never reaches instr.
#[test]
fn test_fts_operator_query_returns_containing_chunk() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(
        &conn,
        "s1",
        "user",
        "authentication middleware configuration",
    );
    insert_chunk_with_embedding(
        &conn,
        1,
        "s1",
        "authentication middleware configuration — mount it before the route handlers",
        1709251200000,
    );

    let results = search(
        &conn,
        "authentication middleware",
        &SearchOptions::default(),
    )
    .unwrap();

    assert_eq!(results.len(), 1);
    let excerpt = &results[0].excerpt;
    assert!(
        excerpt.contains("mount it before the route handlers"),
        "operator query must return the full containing chunk: {excerpt:?}"
    );
}

// T-003 (#8): an FTS hit whose session has no containing qa_chunk falls back to
// the 20-token snippet (with `**` highlight markers), not an empty excerpt.
#[test]
fn test_fts_hit_without_chunk_falls_back_to_snippet() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "authentication flow discussion");
    // no qa_chunk inserted for s1

    let results = search(&conn, "authentication", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    let excerpt = &results[0].excerpt;
    assert!(
        excerpt.contains("**"),
        "fallback must be the FTS snippet (** markers), not a chunk: {excerpt:?}"
    );
}

// T-006 (#8 audit/B): when a session has multiple messages matching the query,
// the chunk of the highest-ranked (most relevant) message is selected, not an
// arbitrary rowid-ordered one. `chunk_match_sql` orders the inner message
// subquery by FTS5 `rank`. The low-rank message is inserted first (lower rowid),
// so a rowid-ordered LIMIT 1 would pick the wrong chunk.
#[test]
fn test_fts_hit_selects_highest_ranked_message_chunk() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    // rowid 1: long, diluted -> lower bm25 rank for "authentication".
    insert_message(
        &conn,
        "s1",
        "user",
        "authentication alpha padded with many extra filler words here to lower its relevance",
    );
    // rowid 2: short, focused -> higher bm25 rank.
    insert_message(&conn, "s1", "user", "authentication beta");
    insert_chunk_with_embedding(
        &conn,
        1,
        "s1",
        "authentication alpha padded with many extra filler words here to lower its relevance — ALPHA_CHUNK_BODY",
        1709251200000,
    );
    insert_chunk_with_embedding(
        &conn,
        2,
        "s1",
        "authentication beta — BETA_CHUNK_BODY",
        1709251200000,
    );

    let results = search(&conn, "authentication", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    let excerpt = &results[0].excerpt;
    assert!(
        excerpt.contains("BETA_CHUNK_BODY"),
        "must select the highest-ranked message's chunk, not the first by rowid: {excerpt:?}"
    );
}

// T-118 (#118): a short matched message text that instr-collides with multiple
// chunks in the same session is ambiguous. The old `instr ... LIMIT 1` returned
// an arbitrary rowid-ordered chunk (here UNRELATED_BODY), which need not be the
// matched message's home chunk. The uniqueness guard (HAVING COUNT(*) = 1) sees
// the collision (count = 2) and falls back to the match-centered snippet, which
// always contains the match and never a wrong chunk.
#[test]
fn test_fts_short_hit_ambiguous_chunk_falls_back_to_snippet() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "yes");
    // Two chunks in the same session both contain the substring "yes" and are
    // equally noise: neither is the matched message's home. rowid 1 (inserted
    // first) is the arbitrary winner of the old `LIMIT 1`.
    insert_chunk_with_embedding(&conn, 1, "s1", "yes — COLLIDING_BODY_A", 1709251200000);
    insert_chunk_with_embedding(&conn, 2, "s1", "yes — COLLIDING_BODY_B", 1709251200000);

    let results = search(&conn, "yes", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    let excerpt = &results[0].excerpt;
    assert!(
        excerpt.contains("**")
            && !excerpt.contains("COLLIDING_BODY_A")
            && !excerpt.contains("COLLIDING_BODY_B"),
        "ambiguous short-text hit must fall back to the match-centered snippet, \
         not either arbitrary colliding chunk: {excerpt:?}"
    );
}

// T-118b (#118): the count=1 acceptance side of the `HAVING COUNT(*) = 1` guard.
// A session with several chunks where exactly one contains the matched text must
// still return that whole chunk (not the snippet). T-001/T-002 only have a single
// chunk, so they pass even if the guard over-suppressed multi-chunk count=1; this
// pins the guard's lower boundary against a regression like `HAVING COUNT(*) > 1`.
#[test]
fn test_fts_hit_returns_unique_chunk_among_multiple() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "needle phrase");
    // Only chunk 1 contains "needle phrase"; chunk 2 is unrelated noise, so the
    // matched text instr-matches exactly one chunk → count=1.
    insert_chunk_with_embedding(&conn, 1, "s1", "needle phrase — HOME_BODY", 1709251200000);
    insert_chunk_with_embedding(&conn, 2, "s1", "unrelated topic NOISE_BODY", 1709251200000);

    let results = search(&conn, "needle", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    let excerpt = &results[0].excerpt;
    assert!(
        excerpt.contains("HOME_BODY") && !excerpt.contains("**"),
        "count=1 must return the full home chunk, not over-suppress to a snippet: {excerpt:?}"
    );
}

// T-192a (#192): false ambiguity. A short message "yes" instr-collides with two
// chunks, so the #118 instr guard saw COUNT(*) = 2 and degraded to a snippet.
// The rowid-range JOIN resolves it: the "yes" message's rowid falls inside
// exactly one group's [lo, hi], so the primary SQL's COUNT(*) = 1 holds and the
// full owning chunk is returned. Perspective: Combination (instr collision ×
// rowid scoping) + Hazard (short-token false ambiguity). RED on current code:
// today's instr-only SQL has no src_rowid columns to JOIN, so this compiles
// against the intended API only and, once Green, must return the full chunk.
#[test]
fn test_192a_short_hit_rowid_range_disambiguates_to_full_chunk() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    // rowid 1: the matched user message "yes" — it lives in the group below.
    insert_message(&conn, "s1", "user", "yes");
    // rowid 2: an unrelated later message whose chunk also contains "yes" as a
    // substring (instr-collision), but whose rowid range excludes rowid 1.
    insert_message(&conn, "s1", "user", "did it say yes or no");
    let yes_rowid = message_rowid(&conn, "s1", "yes");
    let other_rowid = message_rowid(&conn, "s1", "did it say yes or no");

    // Precondition: the FTS top-hit for "yes" must be the "yes" message itself.
    // The primary SQL JOINs the single top-ranked rowid, so if a rank inversion
    // ever makes the longer message win, this fixture (not production) is wrong.
    // Pinning it here makes a Green-time inversion fail loudly on the fixture.
    let top_hit_rowid: i64 = conn
        .query_row(
            "SELECT rowid FROM messages WHERE messages MATCH ?1 AND session_id = ?2 ORDER BY rank LIMIT 1",
            ("yes", "s1"),
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        top_hit_rowid, yes_rowid,
        "fixture precondition: the FTS top-hit for 'yes' is the 'yes' message"
    );

    // Owning group: range covers rowid 1 only → the "yes" hit lands here.
    insert_chunk_with_rowid_range(
        &conn,
        1,
        "s1",
        "yes — OWNER_BODY confirming the deploy",
        1709251200000,
        yes_rowid,
        yes_rowid,
    );
    // Collider: substring "yes" matches via instr, but its range is the later
    // message, which does not contain rowid 1.
    insert_chunk_with_rowid_range(
        &conn,
        2,
        "s1",
        "did it say yes or no — COLLIDER_BODY",
        1709251200000,
        other_rowid,
        other_rowid,
    );

    let results = search(&conn, "yes", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    let excerpt = &results[0].excerpt;
    assert!(
        excerpt.contains("OWNER_BODY") && !excerpt.contains("COLLIDER_BODY"),
        "rowid range must scope the hit to its owning chunk and return it full, \
         not degrade to a snippet under instr collision: {excerpt:?}"
    );
    assert!(
        !excerpt.contains("**"),
        "the disambiguated hit returns a full chunk, not the snippet fallback: {excerpt:?}"
    );
}

// T-192b (#192): assistant-message coverage. The FTS top-hit is an assistant
// message whose rowid is the group's `hi` (user rowid + 1). The range JOIN
// (lo <= rowid <= hi) still matches the owning chunk, so the full chunk is
// returned. This pins that the link is a RANGE, not a single user rowid: a
// user-rowid-only design (lo == hi == user rowid) would exclude the assistant
// rowid and return 0 rows → wrong snippet fallback. Perspective: Boundary (hit
// at the upper bound of the range) + Hazard (assistant-rowid regression).
#[test]
fn test_192b_assistant_hit_at_range_hi_returns_full_chunk() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    // rowid 1 user, rowid 2 assistant. The query word "quasar" appears ONLY in
    // the assistant message, so the assistant message is the unambiguous FTS hit.
    insert_message(&conn, "s1", "user", "what is the brightest object");
    insert_message(&conn, "s1", "assistant", "a quasar outshines its galaxy");
    let user_rowid = message_rowid(&conn, "s1", "what is the brightest object");
    let assistant_rowid = message_rowid(&conn, "s1", "a quasar outshines its galaxy");
    assert_eq!(
        assistant_rowid,
        user_rowid + 1,
        "fixture precondition: assistant rowid is the group's hi"
    );

    // The chunk spans [user_rowid, assistant_rowid]. The hit (assistant) sits at hi.
    insert_chunk_with_rowid_range(
        &conn,
        1,
        "s1",
        "what is the brightest object\na quasar outshines its galaxy — QUASAR_BODY",
        1709251200000,
        user_rowid,
        assistant_rowid,
    );

    let results = search(&conn, "quasar", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    let excerpt = &results[0].excerpt;
    assert!(
        excerpt.contains("QUASAR_BODY") && !excerpt.contains("**"),
        "an assistant hit at the range hi must return the full owning chunk; a \
         user-rowid-only link would miss it and fall to a snippet: {excerpt:?}"
    );
}

// T-192c (#192): the uniqueness guard stays. One user message split into N
// chunks shares the same [lo, hi] range, so a hit rowid in that range matches
// all N chunks → COUNT(*) = N > 1 → HAVING COUNT(*) = 1 rejects → the
// match-centered snippet is returned, never an arbitrary one of the N siblings.
// Perspective: Combination (split group × range JOIN) + Boundary (COUNT just
// above 1). This is the honest-ambiguity case the range link must NOT paper over.
#[test]
fn test_192c_split_group_shared_range_falls_back_to_snippet() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "explain consensus");
    let user_rowid = message_rowid(&conn, "s1", "explain consensus");

    // Two sibling chunks from the same split group: identical [lo, hi] covering
    // the user rowid. Both contain "consensus" → both match the JOIN → COUNT = 2.
    insert_chunk_with_rowid_range(
        &conn,
        1,
        "s1",
        "explain consensus — SPLIT_BODY_A part one",
        1709251200000,
        user_rowid,
        user_rowid,
    );
    insert_chunk_with_rowid_range(
        &conn,
        2,
        "s1",
        "explain consensus — SPLIT_BODY_B part two",
        1709251200000,
        user_rowid,
        user_rowid,
    );

    let results = search(&conn, "consensus", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    let excerpt = &results[0].excerpt;
    assert!(
        excerpt.contains("**")
            && !excerpt.contains("SPLIT_BODY_A")
            && !excerpt.contains("SPLIT_BODY_B"),
        "a hit matching N sibling chunks must reject (COUNT>1) and return the \
         match-centered snippet, not an arbitrary sibling: {excerpt:?}"
    );
}

// T-192d (#192): backward compatibility. A legacy chunk with NULL src_rowid_lo
// is excluded from the primary rowid SQL (WHERE c.src_rowid_lo IS NOT NULL), so
// the hit falls through to the NULL-scoped instr guard and the chunk is still
// returned in full — the pre-#192 behavior. Perspective: Branch (the IS NULL
// fallback path) + Error (NULL input to the range JOIN). This must insert NULL
// lo/hi chunks explicitly; the existing T-001/002/003 only incidentally cover it.
#[test]
fn test_192d_null_rowid_range_falls_back_to_instr_guard() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "legacy authentication question");
    // Legacy chunk: src_rowid_lo / src_rowid_hi default to NULL (helper does not
    // set them), so the primary rowid SQL skips it and the instr guard answers.
    insert_chunk_with_embedding(
        &conn,
        1,
        "s1",
        "legacy authentication question — LEGACY_BODY answered before #192",
        1709251200000,
    );

    let results = search(&conn, "authentication", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    let excerpt = &results[0].excerpt;
    assert!(
        excerpt.contains("LEGACY_BODY") && !excerpt.contains("**"),
        "a NULL-rowid legacy chunk must still resolve via the instr fallback and \
         return the full chunk, preserving pre-#192 behavior: {excerpt:?}"
    );
}

// T-192g (#192): round-trip wiring. The other T-192 tests stamp lo/hi by hand via
// `insert_chunk_with_rowid_range`, so none exercise the real `index_chunks` INSERT
// (indexer.rs) end-to-end into a search result. This drives that path: index two
// messages, then search, and assert the full chunk (user + assistant) comes back
// via the rowid JOIN, not a snippet. A param-order swap of ?4/?5 (lo/hi) in the
// index_chunks INSERT would store an inverted range; with lo > hi the BETWEEN JOIN
// matches nothing and the excerpt degrades to a snippet, which this test catches.
// Perspective: Hazard (param-order swap) + integration seam (chunker→indexer→search).
#[test]
fn test_192g_indexed_chunk_resolves_full_chunk_via_rowid_join() {
    let (_dir, mut conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    // "perihelion" appears only in the user message (the assistant body avoids the
    // word so the trigram FTS hit is unambiguous) → the sole hit rowid sits at the
    // group's lo. The assistant body must still come back with it, proving the full
    // owning chunk was resolved across the [lo, hi] range rather than a match snippet.
    insert_message(&conn, "s1", "user", "define perihelion precisely");
    insert_message(
        &conn,
        "s1",
        "assistant",
        "the closest orbital point — APSIS_REPLY_BODY",
    );

    let stats = index_chunks(&mut conn, None).unwrap();
    assert_eq!(
        stats.chunks_created, 1,
        "one user+assistant pair → one chunk"
    );

    let results = search(&conn, "perihelion", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    let excerpt = &results[0].excerpt;
    assert!(
        excerpt.contains("APSIS_REPLY_BODY") && !excerpt.contains("**"),
        "an index_chunks-written chunk must resolve to the full user+assistant body \
         via the rowid JOIN; an inverted lo/hi range would degrade to a snippet: \
         {excerpt:?}"
    );
}

#[test]
fn test_search_no_match() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "hello world");

    let results = search(&conn, "nonexistent_term_xyz", &SearchOptions::default()).unwrap();

    assert!(results.is_empty());
}

#[test]
fn test_recency_boost_favors_newer() {
    let (_dir, conn) = setup_test_db();

    let now_ms = 1_750_000_000_000_i64; // fixed reference point

    insert_session(&conn, "old", "claude", "/proj", now_ms - 365 * MS_PER_DAY);
    insert_message(&conn, "old", "user", "testing the search feature");

    insert_session(&conn, "new", "claude", "/proj", now_ms);
    insert_message(&conn, "new", "user", "testing the search feature");

    let results = search(
        &conn,
        "testing search",
        &SearchOptions {
            now_ms: Some(now_ms),
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].session.session_id, "new");
}

#[test]
fn test_project_filter() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/home/me/proj-a", 1709251200000);
    insert_message(&conn, "s1", "user", "error handling patterns");
    insert_session(&conn, "s2", "claude", "/home/me/proj-b", 1709251200000);
    insert_message(&conn, "s2", "user", "error handling patterns");

    let results = search(
        &conn,
        "error handling",
        &SearchOptions {
            project: Some("/home/me/proj-a".to_owned()),
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session.project, "/home/me/proj-a");
}

#[test]
fn test_source_filter() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "debugging tips");
    insert_session(&conn, "s2", "codex", "/proj", 1709251200000);
    insert_message(&conn, "s2", "user", "debugging tips");

    let results = search(
        &conn,
        "debugging",
        &SearchOptions {
            source: Some(Source::Codex),
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session.source, Source::Codex);
}

#[test]
fn test_days_filter() {
    let (_dir, conn) = setup_test_db();

    let now_ms = 1_750_000_000_000_i64; // fixed reference point

    insert_session(&conn, "recent", "claude", "/proj", now_ms - MS_PER_DAY); // 1 day ago
    insert_message(&conn, "recent", "user", "rust compiler optimization");
    insert_session(&conn, "old", "claude", "/proj", now_ms - 60 * MS_PER_DAY); // 60 days ago
    insert_message(&conn, "old", "user", "rust compiler optimization");

    let results = search(
        &conn,
        "rust compiler",
        &SearchOptions {
            days: Some(7),
            now_ms: Some(now_ms),
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session.session_id, "recent");
}

#[test]
fn test_project_filter_escapes_wildcards() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/home/me/proj-a", 1709251200000);
    insert_message(&conn, "s1", "user", "wildcard test content");
    insert_session(&conn, "s2", "claude", "/other/path", 1709251200000);
    insert_message(&conn, "s2", "user", "wildcard test content");

    let results = search(
        &conn,
        "wildcard",
        &SearchOptions {
            project: Some("%".to_owned()),
            ..Default::default()
        },
    )
    .unwrap();

    assert!(results.is_empty());
}

#[test]
fn test_cjk_japanese_2char() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "型安全についての議論");

    let results = search(&conn, "型安", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session.session_id, "s1");
    assert!(!results[0].excerpt.is_empty());
}

#[test]
fn test_cjk_single_char() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "Rust言語の型システム");

    let results = search(&conn, "型", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
}

#[test]
fn test_cjk_no_match() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "hello world");

    let results = search(&conn, "没有", &SearchOptions::default()).unwrap();

    assert!(results.is_empty());
}

#[test]
fn test_mixed_cjk_ascii_query() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "Rustの型安全について");
    insert_session(&conn, "s2", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s2", "user", "Pythonのパフォーマンス");

    let results = search(&conn, "Rust 型安全", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session.session_id, "s1");
}

#[test]
fn test_cjk_with_project_filter() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/home/me/proj-a", 1709251200000);
    insert_message(&conn, "s1", "user", "型安全の設計");
    insert_session(&conn, "s2", "claude", "/home/me/proj-b", 1709251200000);
    insert_message(&conn, "s2", "user", "型安全の設計");

    let results = search(
        &conn,
        "型安",
        &SearchOptions {
            project: Some("/home/me/proj-a".to_owned()),
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session.project, "/home/me/proj-a");
}

#[test]
fn test_cjk_recency_favors_newer() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;

    insert_session(&conn, "old", "claude", "/proj", now_ms - 365 * MS_PER_DAY);
    insert_message(&conn, "old", "user", "型安全についての議論");

    insert_session(&conn, "new", "claude", "/proj", now_ms);
    insert_message(&conn, "new", "user", "型安全の設計パターン");

    let results = search(
        &conn,
        "型安全",
        &SearchOptions {
            now_ms: Some(now_ms),
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].session.session_id, "new");
}

#[test]
fn test_cjk_recency_with_candidate_overflow() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;
    let limit = 5;
    let candidate_limit = limit * CANDIDATE_LIMIT_MULTIPLIER; // 15

    // Insert 20 old sessions (exceeds candidate_limit)
    for i in 0..20 {
        let sid = format!("old-{i:02}");
        let ts = now_ms - (365 - i) * MS_PER_DAY;
        insert_session(&conn, &sid, "claude", "/proj", ts);
        insert_message(&conn, &sid, "user", &format!("型安全の議論 パート{i}"));
    }

    // Insert the newest session last (highest rowid)
    insert_session(&conn, "newest", "claude", "/proj", now_ms);
    insert_message(&conn, "newest", "user", "型安全の最新設計");

    let results = search(
        &conn,
        "型安全",
        &SearchOptions {
            limit,
            now_ms: Some(now_ms),
            ..Default::default()
        },
    )
    .unwrap();

    assert!(
        results.len() <= limit,
        "should respect limit: got {} results",
        results.len()
    );
    assert_eq!(
        results[0].session.session_id, "newest",
        "newest session must appear first despite {} candidates exceeding candidate_limit {}",
        21, candidate_limit
    );
}

#[test]
fn test_korean_search() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "한국어 테스트 메시지입니다");

    let results = search(&conn, "테스트", &SearchOptions::default()).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session.session_id, "s1");
}

#[test]
fn test_unbalanced_quote_auto_balanced() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "hello world");

    let result = search(&conn, "\"hello", &SearchOptions::default());
    assert!(
        result.is_ok(),
        "auto-balanced quote should not cause FTS5 error"
    );
}

#[test]
fn test_slash_command_query_matches_body() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "used /challenge to review the plan");

    let results = search(&conn, "/challenge", &SearchOptions::default()).unwrap();
    assert_eq!(
        results.len(),
        1,
        "quoted slash token should match body text"
    );
    assert_eq!(results[0].session.session_id, "s1");
}

#[test]
fn test_score_and_sort_respects_limit() {
    let now_ms = 1_750_000_000_000_i64;
    let candidates: Vec<_> = (0..5)
        .map(|i| (-1.0, make_result(&format!("s{i}"), now_ms)))
        .collect();
    let results = score_and_sort(candidates, now_ms, 3);
    assert_eq!(results.len(), 3);
}

// T-119: the recency blend amplifies negative bm25 ranks, so equal-rank
// candidates sort recent-first under the ascending sort.
#[test]
fn test_score_and_sort_prefers_recent_on_equal_rank() {
    let now_ms = 1_750_000_000_000_i64;
    // exactly 3 half-lives back (3 x RECENCY_HALF_LIFE_DAYS), decay = 0.125
    let old_ts = now_ms - 90 * MS_PER_DAY;
    let candidates = vec![
        (-1.0, make_result("old", old_ts)),
        (-1.0, make_result("recent", now_ms)),
    ];
    let results = score_and_sort(candidates, now_ms, 10);
    assert_eq!(results[0].session.session_id, "recent");
    assert_eq!(results[1].session.session_id, "old");
}

// T-119b: a None-timestamp candidate gets recency_decay = 0.0, so the
// blend is a no-op and it sorts after an equal-rank dated candidate.
// None timestamps are production-reachable (sessions whose JSONL has no
// parseable ISO timestamp line are stored with a NULL timestamp).
#[test]
fn test_score_and_sort_sorts_none_timestamp_last_on_equal_rank() {
    let now_ms = 1_750_000_000_000_i64;
    let mut no_ts = make_result("no-ts", now_ms);
    no_ts.session.timestamp = None;
    let candidates = vec![(-1.0, no_ts), (-1.0, make_result("dated", now_ms))];
    let results = score_and_sort(candidates, now_ms, 10);
    assert_eq!(results[0].session.session_id, "dated");
    assert_eq!(results[1].session.session_id, "no-ts");
}

#[test]
fn test_search_limit_zero_returns_empty() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "hello world");

    let results = search(
        &conn,
        "hello",
        &SearchOptions {
            limit: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_index_then_search_roundtrip() {
    use crate::indexer::{IndexOptions, index_from_dirs};

    let (_dir, mut conn) = setup_test_db();
    let tmp = tempfile::TempDir::new().unwrap();

    let claude_dir = tmp.path().join("claude_projects");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::write(
            claude_dir.join("sess-alpha.jsonl"),
            concat!(
                r#"{"type":"user","cwd":"/home/me/alpha","slug":"alpha-session","message":{"role":"user","content":"how to implement authentication in Rust"},"timestamp":"2026-03-01T12:00:00Z"}"#,
                "\n",
                r#"{"type":"assistant","message":{"role":"assistant","content":"Use the argon2 crate for password hashing"},"timestamp":"2026-03-01T12:00:01Z"}"#,
            ),
        ).unwrap();
    fs::write(
            claude_dir.join("sess-beta.jsonl"),
            r#"{"type":"user","cwd":"/home/me/beta","slug":"beta-session","message":{"role":"user","content":"explain quicksort algorithm"},"timestamp":"2026-03-02T10:00:00Z"}"#,
        ).unwrap();

    let codex_dir = tmp.path().join("codex_sessions");

    let stats = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
    )
    .unwrap();
    assert_eq!(stats.indexed, 2);
    assert_eq!(stats.total_sessions, 2);

    let results = search(&conn, "authentication", &SearchOptions::default()).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session.session_id, "sess-alpha");
    assert_eq!(results[0].session.project, "/home/me/alpha");
    assert!(!results[0].excerpt.is_empty());

    let results = search(&conn, "quicksort", &SearchOptions::default()).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session.session_id, "sess-beta");

    // Project filter — alpha-only. rurico quotes FTS5 operators as literal
    // terms, so test each side separately instead of a single OR query.
    let results = search(
        &conn,
        "authentication",
        &SearchOptions {
            project: Some("/home/me/alpha".to_owned()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session.project, "/home/me/alpha");

    let results = search(
        &conn,
        "algorithm",
        &SearchOptions {
            project: Some("/home/me/alpha".to_owned()),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        results.is_empty(),
        "algorithm lives in beta; alpha filter should exclude it"
    );

    let stats2 = index_from_dirs(
        &mut conn,
        &IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
    )
    .unwrap();
    assert_eq!(stats2.indexed, 0);
    assert_eq!(stats2.total_sessions, 2);

    let results = search(&conn, "authentication", &SearchOptions::default()).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_column_filter_blocked_in_search() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "some content about admin");

    let results = search(&conn, "role:user", &SearchOptions::default()).unwrap();
    assert!(
        results.is_empty(),
        "role:user should not match as column filter"
    );
}

#[test]
fn test_012_search_without_embedder_fts5_only() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(&conn, "s1", "user", "authentication flow discussion");

    let results =
        search_with_embedder(&conn, "authentication", &SearchOptions::default(), None).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session.session_id, "s1");
}

#[test]
fn test_hybrid_search_merges_fts_and_vector() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;

    // s1: matches FTS for "authentication"
    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_message(&conn, "s1", "user", "authentication flow discussion");

    // s2: no FTS match, but has a vec_chunk with embedding close to query
    insert_session(&conn, "s2", "claude", "/proj", now_ms);
    insert_message(&conn, "s2", "user", "unrelated topic about weather");

    insert_chunk_with_embedding(&conn, 1, "s2", "authentication", now_ms);

    assert!(has_vec_data(&conn));

    let embedder = MockEmbedder::new();
    let results = search_with_embedder(
        &conn,
        "authentication",
        &SearchOptions {
            now_ms: Some(now_ms),
            ..Default::default()
        },
        Some(&embedder),
    )
    .unwrap();

    let ids: Vec<&str> = results
        .iter()
        .map(|r| r.session.session_id.as_str())
        .collect();
    assert!(
        ids.contains(&"s1"),
        "s1 should appear (FTS match), got: {ids:?}"
    );
    assert!(
        ids.contains(&"s2"),
        "s2 should appear (vector match), got: {ids:?}"
    );

    // #36: a vector-only hit must expose the matched chunk content, not an empty
    // excerpt — `messages MATCH fts_query` returns no row for these sessions.
    let s2 = results
        .iter()
        .find(|r| r.session.session_id == "s2")
        .unwrap();
    assert!(
        s2.excerpt.contains("authentication"),
        "vector-only excerpt must contain matched chunk content, got: {:?}",
        s2.excerpt
    );
}

#[test]
fn test_short_query_runs_vector_search_when_fts_has_no_trigram_term() {
    // #164: a 2-char query ("ui") has no trigram-indexable FTS term, so
    // `clean_for_trigram` yields None. The bug returned early there, skipping vector
    // search even with embeddings present. The fix lets the vector path answer.
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;

    // s1: an FTS-indexed message with no word/trigram starting "ui", so the query
    // cannot expand to a trigram term and cannot match s1 via FTS.
    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_message(&conn, "s1", "user", "completely separate gardening notes");

    // s2: no FTS message, only a vec_chunk whose content equals the query, so the
    // MockEmbedder produces an exact-match vector.
    insert_session(&conn, "s2", "claude", "/proj", now_ms);
    insert_chunk_with_embedding(&conn, 1, "s2", "ui", now_ms);

    // Precondition: the query produces no trigram FTS query (drives the #164 branch).
    let matched = prepare_match_query(
        &conn,
        "ui",
        "messages_vocab",
        &QueryNormalizationConfig::disabled(),
    )
    .unwrap();
    assert!(
        clean_for_trigram(&matched).is_none(),
        "test precondition: short query must yield no trigram FTS query"
    );

    let opts = SearchOptions {
        now_ms: Some(now_ms),
        ..Default::default()
    };

    // FTS-only (no embedder): no trigram term means no FTS candidate, so empty.
    // This is the pre-fix behavior, preserved when no embeddings exist.
    let fts_only = search(&conn, "ui", &opts).unwrap();
    assert!(
        fts_only.is_empty(),
        "FTS-only short query must return no results, got: {:?}",
        fts_only
            .iter()
            .map(|r| r.session.session_id.as_str())
            .collect::<Vec<_>>()
    );

    // With an embedder, vector search now runs and finds s2 by meaning.
    let embedder = MockEmbedder::new();
    let results = search_with_embedder(&conn, "ui", &opts, Some(&embedder)).unwrap();
    let ids: Vec<&str> = results
        .iter()
        .map(|r| r.session.session_id.as_str())
        .collect();
    assert!(
        ids.contains(&"s2"),
        "short query must reach vector search and find s2, got: {ids:?}"
    );

    // The vec-only hit exposes its chunk content, not an empty excerpt — the
    // fts_query=None path skips the MATCH snippet tiers and uses the vec chunk.
    let s2 = results
        .iter()
        .find(|r| r.session.session_id == "s2")
        .unwrap();
    assert!(
        s2.excerpt.contains("ui"),
        "vector-only excerpt must contain chunk content, got: {:?}",
        s2.excerpt
    );
}

#[test]
fn test_empty_and_whitespace_queries_return_empty_even_with_embedder_and_vec_data() {
    // #164 constraint: "truly empty or whitespace-only queries should still return
    // no results." Removing the short-query early return must not break this, even
    // though an embedder and vec data are present (the condition that now lets short
    // queries through to the vector path). These two cases are caught earlier by the
    // NoSearchableTerms / EmptyInput arm, before the vector fall-through.
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;

    // A vec chunk so has_vec_data() is true and the embedder branch is taken.
    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_chunk_with_embedding(&conn, 1, "s1", "some indexed content", now_ms);

    let opts = SearchOptions {
        now_ms: Some(now_ms),
        ..Default::default()
    };
    let embedder = MockEmbedder::new();

    for query in ["", "   "] {
        let results = search_with_embedder(&conn, query, &opts, Some(&embedder)).unwrap();
        assert!(
            results.is_empty(),
            "query {query:?} must return no results, got: {:?}",
            results
                .iter()
                .map(|r| r.session.session_id.as_str())
                .collect::<Vec<_>>()
        );
    }

    // Deliberate scope decision (#164): a non-empty query with no trigram term —
    // including punctuation-only — DOES reach vector search. The issue scopes the
    // fix as "let vector search run for non-empty queries"; the pre-fix empty result
    // for "..." was an incidental effect of the removed early return, not a promise.
    // Pinned so a future "re-fix" that re-adds an empty-on-punctuation gate is caught.
    let punct = search_with_embedder(&conn, "...", &opts, Some(&embedder)).unwrap();
    assert!(
        !punct.is_empty(),
        "non-empty punctuation-only query must reach vector search, not return empty"
    );
}

#[test]
fn test_vector_only_excerpt_uses_closest_chunk() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;

    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_message(&conn, "s1", "user", "authentication flow discussion");

    // s2 is vector-only with two chunks. The query embedding is closest to
    // chunk 2, so the excerpt must come from chunk 2 (MIN distance), not chunk 1.
    // This pins the bare-column / single-MIN behavior, not just "non-empty".
    insert_session(&conn, "s2", "claude", "/proj", now_ms);
    insert_message(&conn, "s2", "user", "unrelated weather topic");
    insert_chunk_with_embedding(&conn, 1, "s2", "totally unrelated weather forecast", now_ms);
    insert_chunk_with_embedding(&conn, 2, "s2", "authentication", now_ms);

    let embedder = MockEmbedder::new();
    let results = search_with_embedder(
        &conn,
        "authentication",
        &SearchOptions {
            now_ms: Some(now_ms),
            ..Default::default()
        },
        Some(&embedder),
    )
    .unwrap();

    let s2 = results
        .iter()
        .find(|r| r.session.session_id == "s2")
        .unwrap();
    assert!(
        s2.excerpt.contains("authentication"),
        "excerpt must come from the closest chunk (chunk 2), got: {:?}",
        s2.excerpt
    );
    assert!(
        !s2.excerpt.contains("weather"),
        "excerpt must be the MIN-distance chunk, not an arbitrary one, got: {:?}",
        s2.excerpt
    );
}

#[test]
fn test_hybrid_recency_boost_favors_newer() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;

    // Two sessions with identical FTS-matching text and identical embedding content.
    // Only the timestamp differs; hybrid path must rank the newer one first.
    for (sid, ts) in [("old", now_ms - 365 * MS_PER_DAY), ("new", now_ms)] {
        insert_session(&conn, sid, "claude", "/proj", ts);
        insert_message(&conn, sid, "user", "authentication flow discussion");
    }

    for (chunk_id, sid) in [(1_i64, "old"), (2, "new")] {
        insert_chunk_with_embedding(
            &conn,
            chunk_id,
            sid,
            "authentication flow discussion",
            now_ms,
        );
    }

    let embedder = MockEmbedder::new();
    let results = search_with_embedder(
        &conn,
        "authentication",
        &SearchOptions {
            now_ms: Some(now_ms),
            ..Default::default()
        },
        Some(&embedder),
    )
    .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0].session.session_id, "new",
        "hybrid path must apply recency boost so newer session ranks first"
    );
}

#[test]
fn test_hybrid_search_falls_back_on_vector_error() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;

    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_message(&conn, "s1", "user", "authentication flow discussion");

    // Need vec_chunks to trigger hybrid path
    insert_chunk_with_embedding(&conn, 1, "s1", "content", now_ms);

    let embedder = MockEmbedder::failing_after(0);
    let results = search_with_embedder(
        &conn,
        "authentication",
        &SearchOptions {
            now_ms: Some(now_ms),
            ..Default::default()
        },
        Some(&embedder),
    )
    .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session.session_id, "s1");
}

fn hybrid_search_ids(
    conn: &Connection,
    embedder: &MockEmbedder,
    opts: &SearchOptions,
) -> Vec<String> {
    search_with_embedder(conn, "authentication", opts, Some(embedder))
        .unwrap()
        .into_iter()
        .map(|r| r.session.session_id)
        .collect()
}

fn set_session_type(conn: &Connection, sid: &str, session_type: &str) {
    conn.execute(
        "UPDATE sessions SET session_type = ?2 WHERE session_id = ?1",
        rusqlite::params![sid, session_type],
    )
    .unwrap();
}

fn fts_ids(conn: &Connection, opts: &SearchOptions) -> Vec<String> {
    search(conn, "authentication", opts)
        .unwrap()
        .into_iter()
        .map(|r| r.session.session_id)
        .collect()
}

// T-005 (#24/FR-005,FR-006) + T-006 (#24/FR-011): default search excludes an
// automated session but keeps a NULL-session_type (unclassified) one — NULL is
// treated as interactive (COALESCE), so pre-classification sessions never vanish.
#[test]
fn test_default_search_excludes_automated_keeps_null() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "auto", "claude", "/proj", 1709251200000);
    insert_message(&conn, "auto", "user", "authentication flow");
    set_session_type(&conn, "auto", "automated");
    // "null" keeps its NULL session_type (insert_session does not set it).
    insert_session(&conn, "null", "claude", "/proj", 1709251200000);
    insert_message(&conn, "null", "user", "authentication flow");

    let ids = fts_ids(&conn, &SearchOptions::default());

    assert!(
        !ids.contains(&"auto".to_owned()),
        "automated session must be excluded by default: {ids:?}"
    );
    assert!(
        ids.contains(&"null".to_owned()),
        "NULL session_type must stay visible (interactive): {ids:?}"
    );
}

// T-008 (#24/FR-007): include_automated surfaces automated sessions.
#[test]
fn test_include_automated_surfaces_automated() {
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "auto", "claude", "/proj", 1709251200000);
    insert_message(&conn, "auto", "user", "authentication flow");
    set_session_type(&conn, "auto", "automated");

    let ids = fts_ids(
        &conn,
        &SearchOptions {
            include_automated: true,
            ..Default::default()
        },
    );

    assert!(
        ids.contains(&"auto".to_owned()),
        "include_automated must surface automated: {ids:?}"
    );
}

// T-007 (#24/FR-006): the automated exclusion applies on the vec path too — an
// automated session reachable only via vector search is dropped by default.
#[test]
fn test_default_hybrid_excludes_automated_on_vec_path() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;
    let embedder = MockEmbedder::new();
    insert_session(&conn, "keep", "claude", "/proj", now_ms);
    insert_message(&conn, "keep", "user", "authentication flow discussion");
    // "auto" is reachable only via the vec path (chunk embedding, no FTS row).
    insert_session(&conn, "auto", "claude", "/proj", now_ms);
    insert_chunk_with_embedding(&conn, 1, "auto", "authentication", now_ms);
    set_session_type(&conn, "auto", "automated");

    let ids = hybrid_search_ids(
        &conn,
        &embedder,
        &SearchOptions {
            now_ms: Some(now_ms),
            ..Default::default()
        },
    );

    assert!(
        !ids.contains(&"auto".to_owned()),
        "automated vec-only session must be excluded by default: {ids:?}"
    );
    assert!(
        ids.contains(&"keep".to_owned()),
        "non-automated session must remain: {ids:?}"
    );
}

// T-CS002: CurrentSession::Exclude drops the invoking session from BOTH the
// FTS and vec candidate paths. The excluded session is seeded into messages
// (FTS hit) and vec_chunks (vec hit); if the filter were applied on only one
// path, the RRF merge would re-union it from the unfiltered path.
// assert_filter_symmetric proves it is absent from the merged output while the
// other session survives. This is the load-bearing both-paths guard for #77.
// `cur` is intentionally in both paths (not vec-only): that way deleting the
// filter call on EITHER builder resurfaces it. The vec path's own liveness is
// independently covered by test_hybrid_search_merges_fts_and_vector.
#[test]
fn test_hybrid_exclude_current_drops_session_from_both_paths() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;
    let embedder = MockEmbedder::new();

    assert_filter_symmetric(
        || {
            insert_session(&conn, "keep", "claude", "/proj", now_ms);
            insert_message(&conn, "keep", "user", "authentication flow discussion");
            // `cur` is the invoking session: it matches on BOTH paths.
            insert_session(&conn, "cur", "claude", "/proj", now_ms);
            insert_message(&conn, "cur", "user", "authentication flow discussion");
            insert_chunk_with_embedding(&conn, 1, "cur", "authentication", now_ms);
            ("keep".to_owned(), "cur".to_owned())
        },
        || {
            hybrid_search_ids(
                &conn,
                &embedder,
                &SearchOptions {
                    current: CurrentSession::Exclude("cur".to_owned()),
                    now_ms: Some(now_ms),
                    ..Default::default()
                },
            )
        },
        || {
            hybrid_search_ids(
                &conn,
                &embedder,
                &SearchOptions {
                    now_ms: Some(now_ms),
                    ..Default::default()
                },
            )
        },
    );
}

// T-CS003: CurrentSession::Only keeps only the invoking session. `keep`
// matches on both paths but must be dropped; `cur` carries an FTS match and
// survives. The `= ?` arm filters both candidate paths (both-paths rationale:
// T-CS002).
#[test]
fn test_hybrid_only_current_keeps_only_invoking_session() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;
    let embedder = MockEmbedder::new();

    assert_filter_symmetric(
        || {
            insert_session(&conn, "cur", "claude", "/proj", now_ms);
            insert_message(&conn, "cur", "user", "authentication flow discussion");
            // `keep` matches on BOTH paths but must be dropped by Only("cur").
            insert_session(&conn, "keep", "claude", "/proj", now_ms);
            insert_message(&conn, "keep", "user", "authentication flow discussion");
            insert_chunk_with_embedding(&conn, 1, "keep", "authentication", now_ms);
            ("cur".to_owned(), "keep".to_owned())
        },
        || {
            hybrid_search_ids(
                &conn,
                &embedder,
                &SearchOptions {
                    current: CurrentSession::Only("cur".to_owned()),
                    now_ms: Some(now_ms),
                    ..Default::default()
                },
            )
        },
        || {
            hybrid_search_ids(
                &conn,
                &embedder,
                &SearchOptions {
                    now_ms: Some(now_ms),
                    ..Default::default()
                },
            )
        },
    );
}

#[test]
fn test_hybrid_project_filter_excludes_vec_only_mismatch() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;
    let embedder = MockEmbedder::new();

    assert_filter_symmetric(
        || {
            insert_session(&conn, "s1", "claude", "/home/me/proj-a", now_ms);
            insert_message(&conn, "s1", "user", "authentication flow discussion");
            insert_session(&conn, "s2", "claude", "/home/me/proj-b", now_ms);
            insert_message(&conn, "s2", "user", "unrelated topic about weather");
            insert_chunk_with_embedding(&conn, 1, "s2", "authentication", now_ms);
            ("s1".to_owned(), "s2".to_owned())
        },
        || {
            hybrid_search_ids(
                &conn,
                &embedder,
                &SearchOptions {
                    project: Some("/home/me/proj-a".to_owned()),
                    now_ms: Some(now_ms),
                    ..Default::default()
                },
            )
        },
        || {
            hybrid_search_ids(
                &conn,
                &embedder,
                &SearchOptions {
                    now_ms: Some(now_ms),
                    ..Default::default()
                },
            )
        },
    );
}

#[test]
fn test_hybrid_days_filter_excludes_vec_only_old() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;
    let embedder = MockEmbedder::new();

    assert_filter_symmetric(
        || {
            insert_session(&conn, "s1", "claude", "/proj", now_ms - MS_PER_DAY);
            insert_message(&conn, "s1", "user", "authentication flow discussion");
            insert_session(&conn, "s2", "claude", "/proj", now_ms - 60 * MS_PER_DAY);
            insert_message(&conn, "s2", "user", "unrelated topic about weather");
            insert_chunk_with_embedding(&conn, 1, "s2", "authentication", now_ms);
            ("s1".to_owned(), "s2".to_owned())
        },
        || {
            hybrid_search_ids(
                &conn,
                &embedder,
                &SearchOptions {
                    days: Some(7),
                    now_ms: Some(now_ms),
                    ..Default::default()
                },
            )
        },
        || {
            hybrid_search_ids(
                &conn,
                &embedder,
                &SearchOptions {
                    now_ms: Some(now_ms),
                    ..Default::default()
                },
            )
        },
    );
}

#[test]
fn test_hybrid_project_filter_case_insensitive_match() {
    // SQLite `LIKE` is ASCII case-insensitive. Both FTS and vec paths now
    // share the same SQL filter, so a case-mismatched project must still
    // match on the vec path.
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;

    insert_session(&conn, "s1", "claude", "/Home/me/Proj-A", now_ms);
    insert_message(&conn, "s1", "user", "authentication flow discussion");
    insert_chunk_with_embedding(&conn, 1, "s1", "authentication", now_ms);

    let embedder = MockEmbedder::new();
    let ids = hybrid_search_ids(
        &conn,
        &embedder,
        &SearchOptions {
            project: Some("/home/me/proj-a".to_owned()),
            now_ms: Some(now_ms),
            ..Default::default()
        },
    );

    assert!(
        ids.iter().any(|id| id == "s1"),
        "case-mismatched project should match (SQLite LIKE parity), got: {ids:?}"
    );
}

#[test]
fn test_hybrid_source_filter_excludes_vec_only_mismatch() {
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;
    let embedder = MockEmbedder::new();

    assert_filter_symmetric(
        || {
            insert_session(&conn, "s1", "claude", "/proj", now_ms);
            insert_message(&conn, "s1", "user", "authentication flow discussion");
            insert_session(&conn, "s2", "codex", "/proj", now_ms);
            insert_message(&conn, "s2", "user", "unrelated topic about weather");
            insert_chunk_with_embedding(&conn, 1, "s2", "authentication", now_ms);
            ("s1".to_owned(), "s2".to_owned())
        },
        || {
            hybrid_search_ids(
                &conn,
                &embedder,
                &SearchOptions {
                    source: Some(Source::Claude),
                    now_ms: Some(now_ms),
                    ..Default::default()
                },
            )
        },
        || {
            hybrid_search_ids(
                &conn,
                &embedder,
                &SearchOptions {
                    now_ms: Some(now_ms),
                    ..Default::default()
                },
            )
        },
    );
}

#[test]
fn test_hybrid_unknown_source_does_not_crowd_out_valid_hit() {
    // Regression: if `fetch_session_metadata` skips a merged hit (e.g. the
    // `sessions.source` value is unknown to `Source::from_db`), that hit
    // must be pruned before `truncate(limit)`, or a higher-rank invalid
    // session could consume the slot a valid one should have used.
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;

    insert_session(&conn, "s_valid", "claude", "/proj", now_ms);
    insert_message(&conn, "s_valid", "user", "authentication flow discussion");

    insert_session(&conn, "s_unknown", "mystery_source_v99", "/proj", now_ms);
    insert_message(&conn, "s_unknown", "user", "unrelated topic");
    insert_chunk_with_embedding(&conn, 1, "s_unknown", "authentication", now_ms);

    let embedder = MockEmbedder::new();
    let results = search_with_embedder(
        &conn,
        "authentication",
        &SearchOptions {
            limit: 1,
            now_ms: Some(now_ms),
            ..Default::default()
        },
        Some(&embedder),
    )
    .unwrap();

    assert_eq!(
        results.len(),
        1,
        "s_valid must fill the limit slot even if s_unknown ranks above it"
    );
    assert_eq!(results[0].session.session_id, "s_valid");
}

#[test]
fn test_search_cjk_short_term_with_special_chars() {
    // The actual bug scenario: 2-char CJK term where vocab contains
    // punctuation-laden trigrams (e.g. "方針：", "方針*", "方針（")
    let (_dir, conn) = setup_test_db();
    insert_session(&conn, "s1", "claude", "/proj", 1709251200000);
    insert_message(
        &conn,
        "s1",
        "user",
        "方針：まとめ 方針*概要 方針（案） ディレクトリ整理の方針",
    );

    // "方針" is 2 chars → triggers vocab expansion with special-char trigrams
    let results = search(&conn, "ディレクトリ整理 方針", &SearchOptions::default())
        .expect("CJK short term + long term should not cause FTS5 error");
    assert_eq!(results.len(), 1);
}

#[test]
fn segment_script_boundaries_splits_multi_script_run() {
    // #167: a no-space natural-language query (助詞 dropped) is a single trigram
    // phrase that only matches contiguously. Inserting a space at each script
    // transition yields the implicit-AND of per-word phrases — the same form the
    // user could have typed with spaces.
    assert_eq!(
        segment_script_boundaries("textlint日本語ドキュメント校正"),
        "textlint 日本語 ドキュメント 校正"
    );
    // han↔hiragana and katakana↔hiragana transitions split too.
    assert_eq!(segment_script_boundaries("校正する"), "校正 する");
    assert_eq!(segment_script_boundaries("Reactのhooks"), "React の hooks");
}

#[test]
fn segment_script_boundaries_keeps_ascii_alnum_whole() {
    // ASCII letter↔digit is NOT a split boundary, so technical tokens survive.
    for q in ["utf8", "v2", "sha256", "h264", "fix the bug"] {
        assert_eq!(
            segment_script_boundaries(q),
            q,
            "ASCII/space query must be byte-identical (precision guard, AC2)"
        );
    }
}

#[test]
fn segment_script_boundaries_is_noop_on_single_script_and_quotes() {
    // A pure-han kanji compound has no script transition; splitting it needs a
    // dictionary (out of scope), so it is left intact.
    assert_eq!(
        segment_script_boundaries("機械学習日本語処理"),
        "機械学習日本語処理"
    );
    // Already space-separated input is unchanged (spaces are not script chars).
    assert_eq!(
        segment_script_boundaries("日本語 ドキュメント"),
        "日本語 ドキュメント"
    );
    // A double-quoted span is an explicit phrase-match intent: leave it verbatim.
    assert_eq!(
        segment_script_boundaries("\"日本語ドキュメント\""),
        "\"日本語ドキュメント\""
    );
}

#[test]
fn degraded_no_space_cjk_query_reaches_session() {
    // #167: in degraded mode (FTS-only, no embedder) a no-space multi-script
    // query must reach the session whose message keeps the 助詞.
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;
    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_message(
        &conn,
        "s1",
        "user",
        "textlintで日本語ドキュメントを校正する",
    );
    // An unrelated session that must not be returned (precision guard).
    insert_session(&conn, "s2", "claude", "/proj", now_ms);
    insert_message(&conn, "s2", "user", "completely unrelated gardening notes");

    let opts = SearchOptions {
        now_ms: Some(now_ms),
        ..Default::default()
    };
    let ids = |q: &str| -> Vec<String> {
        search(&conn, q, &opts)
            .unwrap()
            .into_iter()
            .map(|r| r.session.session_id)
            .collect()
    };

    let no_space = ids("textlint日本語ドキュメント校正");
    assert!(
        no_space.iter().any(|s| s == "s1"),
        "degraded no-space query must reach s1, got: {no_space:?}"
    );
    assert!(
        !no_space.iter().any(|s| s == "s2"),
        "no-space query must not match the unrelated session, got: {no_space:?}"
    );

    // Parity: the no-space form reaches the same sessions as the space-separated
    // form the user could have typed (segmentation produces that exact form).
    let spaced = ids("textlint 日本語 ドキュメント 校正");
    assert_eq!(
        no_space, spaced,
        "no-space query must reach parity with the space-separated form (AC1)"
    );
}
#[test]
fn segment_falls_back_to_raw_when_no_anchor_term() {
    // #167 review Finding 1: segmenting an all-short query (no run ≥3 chars)
    // into multiple vocab-expanded OR-groups can overflow amici's MAX_COMBOS
    // (25×25 > 100) → `fixed_only()` with no fixed term → None → zero results,
    // regressing the contiguous-phrase match the raw query had. The fix keeps
    // the raw query when the split yields no ≥3-char anchor.
    //
    // Only reproduces at corpus scale: the OR-group product must exceed 100, so
    // 校正* and する* each need >10 distinct following chars in the vocab. A
    // small DB keeps the groups tiny and hides the overflow (observer effect).
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;
    // Target: an indexed message whose text contains contiguous "校正する".
    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_message(&conn, "s1", "user", "これを校正する手順をまとめた");
    // Inflate the 校正* and する* trigram vocab past the overflow threshold.
    let following = "案版中後前用例集表記号点字形面"; // 14 distinct trailing chars
    for (i, c) in following.chars().enumerate() {
        insert_session(&conn, &format!("k{i}"), "claude", "/proj", now_ms);
        insert_message(
            &conn,
            &format!("k{i}"),
            "user",
            &format!("校正{c}についてのメモ"),
        );
        insert_session(&conn, &format!("u{i}"), "claude", "/proj", now_ms);
        insert_message(
            &conn,
            &format!("u{i}"),
            "user",
            &format!("する{c}に関する記録"),
        );
    }
    let opts = SearchOptions {
        now_ms: Some(now_ms),
        ..Default::default()
    };
    let r = search(&conn, "校正する", &opts).unwrap();
    assert!(
        r.iter().any(|x| x.session.session_id == "s1"),
        "all-short query must keep raw-phrase recall (no MAX_COMBOS overflow), got: {:?}",
        r.iter()
            .map(|x| x.session.session_id.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn segment_prolonged_sound_mark_does_not_strand_hiragana() {
    // #167 review Finding 2: ー (U+30FC) classifies as Katakana, so a hiragana
    // word like うーん would split into three 1-char runs, all sub-trigram →
    // None → zero results. The no-anchor fallback keeps the raw query.
    assert_eq!(segment_script_boundaries("うーん"), "う ー ん");
    let (_dir, conn) = setup_test_db();
    let now_ms = 1_750_000_000_000_i64;
    insert_session(&conn, "s1", "claude", "/proj", now_ms);
    insert_message(&conn, "s1", "user", "それはうーんと悩ましい問題だ");
    let opts = SearchOptions {
        now_ms: Some(now_ms),
        ..Default::default()
    };
    let r = search(&conn, "うーん", &opts).unwrap();
    assert!(
        r.iter().any(|x| x.session.session_id == "s1"),
        "hiragana word with ー must keep raw recall, got: {:?}",
        r.iter()
            .map(|x| x.session.session_id.as_str())
            .collect::<Vec<_>>()
    );
}
