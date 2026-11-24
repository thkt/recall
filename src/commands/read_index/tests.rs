use super::*;

#[test]
fn an_index_older_than_v0_14_is_refused_as_a_data_error_that_names_both_ways_out() {
    let err = require_current_schema(SchemaState::Unsupported).unwrap_err();

    assert_eq!(
        (err.error_code().code(), err.to_string()),
        (
            65,
            "this index's schema predates recall 0.14.0 and cannot be opened; build a new index \
             with `recall index` at another `--db-path`, or upgrade this one by running \
             `recall classify` from recall 0.16.1 on it"
                .to_owned()
        )
    );
}

#[test]
fn an_index_whose_create_did_not_finish_is_refused_as_a_data_error_that_names_recall_index() {
    let err = require_current_schema(SchemaState::Incomplete).unwrap_err();

    assert_eq!(
        (err.error_code().code(), err.to_string()),
        (
            65,
            "this index was not fully created; run `recall index` to finish it".to_owned()
        )
    );
}
