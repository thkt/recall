/// The note a read from the last checkpoint adds in a directory that refuses new files
/// (`stale_wal_note` in src/db.rs). The db unit tests and the CLI tests compare it whole.
pub const READ_ONLY_DIR_NOTE: &str = "read-only open cannot see uncheckpointed changes in the \
     write-ahead log; the directory is read-only, so copy the DB together with its `-wal` \
     (and `-shm`, if present) sidecar files elsewhere and run `recall index --db-path <copy>` \
     to checkpoint them there.";
