mod query_normalize;
mod search;
pub use query_normalize::QueryNormalizationConfig;
use rusqlite::ffi::{SQLITE_OK, sqlite3, sqlite3_api_routines, sqlite3_auto_extension};
pub use search::{MatchFtsQuery, SanitizeError, prepare_match_query};
use std::{ffi::c_char, mem::transmute, sync::OnceLock};

#[allow(unsafe_code)]
pub fn ensure_sqlite_vec() -> anyhow::Result<()> {
    static REGISTER: OnceLock<i32> = OnceLock::new();
    // SAFETY: sqlite-vec's FFI declaration erases the arguments of sqlite3_vec_init.
    // Its pinned SQLite extension ABI matches the callback signature used below.
    let code = *REGISTER.get_or_init(|| unsafe {
        sqlite3_auto_extension(Some(transmute::<
            *const (),
            unsafe extern "C" fn(
                *mut sqlite3,
                *mut *mut c_char,
                *const sqlite3_api_routines,
            ) -> i32,
        >(sqlite_vec::sqlite3_vec_init as *const ())))
    });
    anyhow::ensure!(code == SQLITE_OK, "sqlite-vec registration failed: {code}");
    Ok(())
}
