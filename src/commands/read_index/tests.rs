use super::*;

#[test]
fn require_current_schema_stale_is_data_error_naming_rebuild() {
    let err = require_current_schema(SchemaState::Stale).unwrap_err();
    assert!(
        matches!(err, RecallError::DataError(_)),
        "stale schema must be a DataError, got {err:?}"
    );
    assert!(
        err.to_string().contains("recall rebuild"),
        "message must name the fix: {err}"
    );
    assert_eq!(err.error_code().code(), 65);
}

#[test]
fn require_current_schema_current_is_ok() {
    assert!(require_current_schema(SchemaState::Current).is_ok());
}

#[test]
fn require_current_schema_empty_is_ok() {
    assert!(require_current_schema(SchemaState::Empty).is_ok());
}
