---
status: "accepted"
date: 2026-11-03
decision-makers: thkt
scope: "vendored llama-cpp-sys-2 build; the llama.cpp submodule pin stays unchanged"
---

# Apply Upstream llama.cpp Fixes as Build-Time Patches on the Pinned Submodule

## Context and Problem Statement

recall pins the llama.cpp submodule at `1b70350c92563449576bcbbf552994b1d1ed847d` because the independent EmbeddingGemma 2 evaluation ran against exactly that pairing ([vendor/README.md](../../vendor/README.md)). A fix that landed upstream after the pin is therefore not in recall's build, and taking it by moving the pin means redoing every validation the README requires.

[Issue #381](https://github.com/thkt/recall/issues/381) hit this case. After #380, the parent's model load is the longest stage of `recall search`. The cause is in llama.cpp: `src/llama-model-loader.cpp` converts all metadata to strings even when logging is discarded, and the vocabulary load converts every element of its 26-thousand-token and 510-thousand-merge arrays to a string. Upstream removed this work in [ggml-org/llama.cpp#30679](https://github.com/ggml-org/llama.cpp/pull/30679) (commit `824eaecf48`, 2026-10-21, 7 files, +73/-37), which the pin (2026-10-13) lacks.

Measurements recorded in Issue #381 (release build, `recall search authentication`, 15 alternating runs each, load average 3 to 4):

| Build                                    | model load p50 | search p50 |
| ---------------------------------------- | -------------- | ---------- |
| main                                     | 335.2 ms       | 0.992 s    |
| One-line patch that skips array printing | 237.3 ms       | 0.887 s    |
| All of #30679                            | 191.9 ms       | 0.825 s    |

The pin cannot move for this, and the repository has no way to carry a change to llama.cpp. This decision sets how recall takes an upstream fix while the pin stays.

## Decision Drivers

- The pairing of binding and llama.cpp revision is the unit that was validated, so the submodule pointer must not move
- A patch that silently fails to apply must stop the build, because a build that is missing the patch looks identical to one that has it
- The tokenizer and the embedding must stay bit-identical to the unpatched build, since recall's index vectors depend on them
- The submodule work tree must stay clean after a build
- A carried patch must disappear when the pin moves past the commit it came from

## Considered Options

- Apply the upstream commit as a patch at build time, onto a copy of llama.cpp in `OUT_DIR` (chosen)
- Move the submodule to a revision that contains #30679
- Fork llama.cpp and add the commit there
- Apply a one-line patch that only skips array printing

## Decision Outcome

Chosen option: "apply the upstream commit as a build-time patch on a copy". The vendored `build.rs` copies llama.cpp into `OUT_DIR`, applies the patches there, writes back only the files whose content changed, and hands that copy to CMake. The submodule and its pointer stay as they are.

Moving the pin was rejected because it spans 452 upstream commits and reopens the full validation in `vendor/README.md`. Forking was rejected because it adds a repository to maintain for one commit. The one-line patch was rejected because it saves about 45 ms less than #30679 and does not touch the vocabulary load.

| Rule                  | Decision                                                                                                                                                                                                                                     |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Source of a patch     | An upstream llama.cpp commit only, taken whole. A change written by recall is not carried as a patch                                                                                                                                         |
| Reach of a patch      | Files below `src/` only. A patch that changes any other path is rejected. bindgen and the cc wrapper keep reading `include/`, `ggml/include` and `common/` from the original tree, so a change there would split the build from its bindings |
| Failure               | A patch that does not apply fails the build. The error names the `.patch` file, because git's own error names only a source file                                                                                                             |
| Proof of application  | After applying, the build reverses the patch as a dry run against the staged copy and again against the synced tree CMake builds from. A skipped patch, which `git apply` reports with exit code 0, fails the build                                                                              |
| Isolation from recall | git runs detached from the recall repository. `OUT_DIR` sits under `target/`, and without the isolation `git apply` would see recall's repository and skip the patch                                                                          |
| Pin                   | The submodule stays at `1b70350`. The work tree stays clean after a build                                                                                                                                                                    |
| Baseline              | `RECALL_LLAMA_SKIP_PATCHES=1` builds without patches and emits a build warning, so an unpatched build can be compared against a patched one                                                                                                  |
| Pin bump              | Moving the pin to or past a patch's upstream commit removes that patch in the same change                                                                                                                                                    |

### Verification policy

The patched build must produce the same token ids, the same vocabulary pieces and the same embedding bits as the unpatched build. The order of the comparison is fixed:

1. Compare two dumps from the unpatched build first (A/A). If they differ, the platform's inference is not reproducible, and an A/B difference would mean nothing.
2. Compare the unpatched dump against the patched dump (A/B) only after A/A matches byte for byte.
3. If A/B differs, stop adopting the patch and return the decision to the user. Do not change `ModelSpec`'s `implementation` to absorb a difference.

When the dumps match, `ModelSpec`'s `implementation` (`recall-gemma-llama-v1`) stays unchanged. The frozen independent vectors are not used for this comparison because their artifacts are not available in the working environment.

The commands and the text set are written in [vendor/README.md](../../vendor/README.md) only. This record does not repeat them.

### Consequences

- Good, because the model load shrinks by the upstream change (335.2 ms to 191.9 ms at p50 in the Issue #381 measurement) without a pin change
- Good, because a patch that fails to apply, applies partly or touches a path outside `src/` stops the build with the patch file name
- Good, because the validated pairing stays recorded as `1b70350`, and the patch list is the whole difference from it
- Bad, because `build.rs` now copies the llama.cpp tree into `OUT_DIR`, which adds build code that did not exist before
- Bad, because the build under test is no longer the pinned source as published, so reproducing it needs the patch files as well as the submodule
- Bad, because a patch is limited to `src/`, so an upstream fix that also changes headers cannot be taken this way

### Confirmation

- `tests/llama_patches.rs` checks that a patch outside `src/` is rejected, that a patch that does not apply fails with its file name even when the staging directory sits inside another git repository, that the source tree is unchanged after applying, and that syncing rewrites only the files whose content differs.
- The `APPLIED_PATCHES` constant exposed by `llama-cpp-sys-2` lists the patches in the linked build, and a recall test asserts that the vocabulary loading patch is among them.
- The real-model comparison follows the policy above, with its steps in [vendor/README.md](../../vendor/README.md).

## More Information

- Issue: [#381](https://github.com/thkt/recall/issues/381)
- Upstream: [ggml-org/llama.cpp#30679](https://github.com/ggml-org/llama.cpp/pull/30679), commit `824eaecf48`
- Related: [DR 0013](0013-skip-the-isolated-model-probe-on-search-in-a-verified-environment.md) (the probe skip that made the parent's model load the longest stage of search)

### Reassessment Triggers

- The pin moves to or past `824eaecf48`; delete the patch and its README entry in that change
- A patch no longer applies after a pin bump; decide between dropping it and regenerating it from the new base
- A second upstream commit is proposed; confirm that it touches `src/` only, and reconsider whether carrying several patches is still cheaper than moving the pin
- The unpatched A/A comparison differs on a platform; the bit-identity check cannot establish parity there
