#[path = "../vendor/llama-cpp-sys-2/patches.rs"]
mod patches;

use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, SystemTime},
};
use tempfile::TempDir;

const EDIT_SRC_PATCH: &str = "\
diff --git a/src/a.txt b/src/a.txt
index 0000000..0000000 100644
--- a/src/a.txt
+++ b/src/a.txt
@@ -1,3 +1,3 @@
 one
-two
+TWO
 three
";

const EDIT_OUTSIDE_SRC_PATCH: &str = "\
diff --git a/ggml/b.txt b/ggml/b.txt
index 0000000..0000000 100644
--- a/ggml/b.txt
+++ b/ggml/b.txt
@@ -1,3 +1,3 @@
 one
-two
+TWO
 three
";

const MISMATCHED_CONTEXT_PATCH: &str = "\
diff --git a/src/a.txt b/src/a.txt
index 0000000..0000000 100644
--- a/src/a.txt
+++ b/src/a.txt
@@ -1,3 +1,3 @@
 one
-not the second line
+TWO
 three
";

const ORIGINAL: &str = "one\ntwo\nthree\n";

fn write(root: &Path, relative: &str, content: &str) {
    let path = root.join(relative);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, content).unwrap();
}

fn read(root: &Path, relative: &str) -> String {
    fs::read_to_string(root.join(relative)).unwrap()
}

fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).unwrap();
        }
    }
}

fn llama_tree(root: &Path) {
    write(root, "src/a.txt", ORIGINAL);
    write(root, "ggml/b.txt", ORIGINAL);
}

fn modified_time(path: &Path) -> SystemTime {
    fs::metadata(path).unwrap().modified().unwrap()
}

fn set_modified_time(path: &Path, time: SystemTime) {
    File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_modified(time)
        .unwrap();
}

#[test]
fn a_patch_that_changes_a_path_outside_src_is_rejected_with_an_error_naming_the_patch_file() {
    let work = TempDir::new().unwrap();
    let staged = work.path().join("staged");
    let patch_dir = work.path().join("patches");
    llama_tree(&staged);
    write(&patch_dir, "0001-outside-src.patch", EDIT_OUTSIDE_SRC_PATCH);

    let error = patches::apply_patches(&staged, &patch_dir)
        .expect_err("a patch touching ggml/ must be rejected");

    assert!(
        error.contains("0001-outside-src.patch"),
        "error does not name the patch file: {error}"
    );
    assert_eq!(read(&staged, "ggml/b.txt"), ORIGINAL);
}

#[test]
fn a_patch_that_does_not_apply_fails_with_an_error_naming_the_patch_file_even_when_the_staging_directory_sits_inside_another_git_repository()
 {
    let work = TempDir::new().unwrap();
    let init = Command::new("git")
        .args(["init", "--quiet"])
        .arg(work.path())
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .output()
        .unwrap();
    assert!(init.status.success(), "git init failed");
    let staged = work.path().join("target/out/llama.cpp");
    let patch_dir = work.path().join("patches");
    llama_tree(&staged);
    write(&patch_dir, "0001-stale-context.patch", MISMATCHED_CONTEXT_PATCH);

    let error = patches::apply_patches(&staged, &patch_dir)
        .expect_err("a patch with mismatched context must fail, not be skipped");

    assert!(
        error.contains("0001-stale-context.patch"),
        "error does not name the patch file: {error}"
    );
    assert_eq!(read(&staged, "src/a.txt"), ORIGINAL);
}

#[test]
fn applying_the_patches_changes_the_staged_copy_and_leaves_the_source_tree_unchanged() {
    let work = TempDir::new().unwrap();
    let source = work.path().join("source");
    let staged = work.path().join("staged");
    let patch_dir = work.path().join("patches");
    llama_tree(&source);
    copy_tree(&source, &staged);
    write(&patch_dir, "0001-edit-src.patch", EDIT_SRC_PATCH);

    let applied = patches::apply_patches(&staged, &patch_dir).unwrap();

    assert_eq!(read(&staged, "src/a.txt"), "one\nTWO\nthree\n");
    assert_eq!(read(&source, "src/a.txt"), ORIGINAL);
    assert_eq!(applied, vec!["0001-edit-src.patch".to_owned()]);
}

#[test]
fn syncing_a_staged_tree_rewrites_only_the_destination_files_whose_content_differs_and_keeps_the_modification_time_of_the_rest()
 {
    let work = TempDir::new().unwrap();
    let staged = work.path().join("staged");
    let dest = work.path().join("dest");
    write(&staged, "src/changed.txt", "new\n");
    write(&staged, "src/same.txt", "same\n");
    write(&staged, "src/added.txt", "added\n");
    write(&dest, "src/changed.txt", "old\n");
    write(&dest, "src/same.txt", "same\n");
    let long_ago = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000_000);
    set_modified_time(&dest.join("src/changed.txt"), long_ago);
    set_modified_time(&dest.join("src/same.txt"), long_ago);

    let mut rewritten = patches::sync_changed(&staged, &dest).unwrap();

    rewritten.sort();
    assert_eq!(
        rewritten,
        vec![PathBuf::from("src/added.txt"), PathBuf::from("src/changed.txt")]
    );
    assert_eq!(read(&dest, "src/changed.txt"), "new\n");
    assert_eq!(read(&dest, "src/added.txt"), "added\n");
    assert_eq!(modified_time(&dest.join("src/same.txt")), long_ago);
    assert!(modified_time(&dest.join("src/changed.txt")) > long_ago);
}

#[test]
fn the_linked_llama_cpp_build_reports_the_vocabulary_loading_backport_among_its_applied_patches() {
    assert!(
        llama_cpp_sys_2::APPLIED_PATCHES.contains(&"0001-llama-30679-faster-vocab-load.patch"),
        "APPLIED_PATCHES = {:?}",
        llama_cpp_sys_2::APPLIED_PATCHES
    );
}
