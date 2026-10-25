use super::*;
use crate::model::hash_file;
use std::{env, time::Duration};

fn unit_vector(v: &[f32]) {
    assert_eq!(v.len(), EMBEDDING_DIMS);
    assert!(v.iter().all(|x| x.is_finite()));
    let norm: f64 = v.iter().map(|x| f64::from(*x).powi(2)).sum();
    assert!((norm - 1.).abs() < 1e-5);
}

#[test]
#[ignore = "requires the pinned public model and tokenizer, explicitly staged by CI"]
fn public_model_contract() {
    let root =
        env::var_os("RECALL_GEMMA_TEST_ARTIFACT_DIR").expect("explicit public model directory");
    let root = Path::new(&root);
    let spec = ModelSpec::default();
    assert_eq!(
        hash_file(&root.join("model.gguf")).unwrap(),
        spec.model_sha256
    );
    assert_eq!(
        hash_file(&root.join("tokenizer.json")).unwrap(),
        spec.tokenizer_sha256
    );
    let gemma = Gemma::load(&root.join("model.gguf"), &root.join("tokenizer.json"), spec).unwrap();
    let query = gemma
        .embed_query("Find the database rollback instructions")
        .unwrap();
    unit_vector(&query);
    let samples = [
        "Keep the previous database for rollback.",
        "A watercolor landscape with an orange sunset.",
    ];
    let references = samples.map(|text| gemma.embed_document(text).unwrap());
    for reference in &references {
        unit_vector(&reference.chunks()[0]);
    }
    assert!(
        gemma.tokenizer.get().is_none(),
        "short inputs must not load the offset tokenizer"
    );

    let texts: Vec<_> = (0..33).map(|i| samples[i % samples.len()]).collect();
    for budget in [None, Some(1)] {
        let inputs = if budget.is_some() {
            &texts[..3]
        } else {
            &texts[..]
        };
        let out = gemma
            .embed_documents_batch_with_options(
                inputs,
                &EmbedOptions {
                    token_budget: budget,
                    forward_pause: Some(Duration::from_millis(1)),
                },
            )
            .unwrap();
        assert_eq!(out.len(), inputs.len());
        for (i, doc) in out.into_iter().enumerate() {
            assert_eq!(doc.chunks().len(), 1);
            unit_vector(&doc.chunks()[0]);
            let max_diff = doc.chunks()[0]
                .iter()
                .zip(&references[i % references.len()].chunks()[0])
                .map(|(a, b)| (a - b).abs())
                .fold(0_f32, f32::max);
            assert!(
                max_diff < 1e-3,
                "batch ownership or attention isolation drift: {max_diff}"
            );
        }
    }
    assert!(gemma.embed_documents_batch(&[]).unwrap().is_empty());
    {
        let model = gemma.model.lock().unwrap();
        assert!(Gemma::encode(&model, &[], &EmbedOptions::default()).is_err());
        for target in [511, 512, 513, 8192, 8193] {
            let text = (target - 40..=target)
                .map(|n| "a ".repeat(n))
                .find(|s| {
                    model
                        .vocab()
                        .tokenize(
                            format!("{}{s}", gemma.spec.query_prefix).as_bytes(),
                            true,
                            false,
                        )
                        .len()
                        == target
                })
                .expect("exact tokenizer boundary");
            let tokens = gemma
                .query_tokens(&model, &text, &gemma.spec.query_prefix)
                .unwrap();
            assert_eq!(tokens.len(), target.min(MAX_TOKENS));
            if target > MAX_TOKENS {
                assert_eq!(tokens.last(), Some(&model.vocab().eos()));
            }
            if target == 513 {
                // Full-length inference is measured on M3; avoid quadratic work on CPU CI.
                unit_vector(
                    &Gemma::encode(&model, &[tokens], &EmbedOptions::default()).unwrap()[0],
                );
            }
        }
        let long = format!("  {} 終端TAIL\n", "日本語🙂 e\u{301} token ".repeat(2500));
        let windows = gemma.document_tokens(&model, &long).unwrap();
        assert!(windows.len() > 1);
        let mut end = 0;
        for window in windows {
            assert!(window.byte_range.start <= end && !window.byte_range.is_empty());
            let text = long.get(window.byte_range.clone()).unwrap();
            assert_eq!(
                window.tokens,
                model.vocab().tokenize(
                    format!("{}{text}", gemma.spec.document_prefix).as_bytes(),
                    true,
                    false
                )
            );
            assert!(window.tokens.len() <= MAX_TOKENS);
            end = end.max(window.byte_range.end);
        }
        assert_eq!(end, long.len());
    }
    assert!(gemma.tokenizer.get().unwrap().is_ok());
    let broken = Gemma {
        model: gemma.model,
        tokenizer: OnceLock::new(),
        tokenizer_path: root.join("missing-tokenizer.json"),
        spec: gemma.spec,
    };
    assert!(broken.embed_document(&"a ".repeat(9000)).is_err());
}
