//! Numeric-only observations for one index invocation. No source text or paths.
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::{self, Write};
#[cfg(test)]
use std::rc::Rc;
use std::time::{Duration, Instant};

type Reporter = Box<dyn Fn(&str)>;

#[derive(Default)]
pub(crate) struct Observer {
    seconds: RefCell<BTreeMap<&'static str, f64>>,
    counts: RefCell<BTreeMap<&'static str, usize>>,
    reporter: Option<Reporter>,
}

impl Observer {
    pub(crate) fn stderr() -> Self {
        Self::with_reporter(|line| {
            let _ = writeln!(io::stderr().lock(), "{line}");
        })
    }

    pub(crate) fn with_reporter(reporter: impl Fn(&str) + 'static) -> Self {
        Self {
            reporter: Some(Box::new(reporter)),
            seconds: RefCell::default(),
            counts: RefCell::default(),
        }
    }

    fn emit(&self, line: &str) {
        if let Some(report) = &self.reporter {
            report(line);
        }
    }

    pub(crate) fn count(&self, key: &'static str, value: usize) {
        self.counts.borrow_mut().insert(key, value);
    }

    pub(crate) fn add_seconds(&self, key: &'static str, seconds: f64) {
        *self.seconds.borrow_mut().entry(key).or_default() += seconds;
    }

    pub(crate) fn seconds(&self, key: &'static str) -> f64 {
        self.seconds.borrow().get(key).copied().unwrap_or_default()
    }

    pub(crate) fn stage(&self, name: &'static str) -> Stage<'_> {
        self.emit(&format!("index: {name}: started (not yet complete)"));
        Stage {
            observer: self,
            name,
            start: Instant::now(),
            last: RefCell::new(Instant::now()),
            finished: false,
        }
    }

    pub(crate) fn start_embedding(&self, pending: usize) {
        self.count("chunks_pending_snapshot", pending);
        self.count("inference_batches", 0);
        self.count("inference_chunks", 0);
        self.count("chunks_saved", 0);
        self.count("chunks_failed", 0);
        self.count("chunks_stale", 0);
        self.count("chunks_save_failed", 0);
        self.count("chunks_unattempted", pending);
        self.count("chunks_remaining_snapshot", pending);
        self.embedding_progress();
    }

    pub(crate) fn embedding_progress(&self) {
        self.emit(&format!(
            "index: embedding counts: {}",
            serde_json::json!(*self.counts.borrow())
        ));
    }

    pub(crate) fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({ "seconds": *self.seconds.borrow(), "counts": *self.counts.borrow() })
    }
}

pub(crate) struct Stage<'a> {
    observer: &'a Observer,
    name: &'static str,
    start: Instant,
    last: RefCell<Instant>,
    finished: bool,
}

impl Stage<'_> {
    /// Uses already-known totals; never scans the DB just to render progress.
    pub(crate) fn progress(&self, processed: usize, total: usize, saved: usize) {
        if processed != total && self.last.borrow().elapsed() < Duration::from_secs(1) {
            return;
        }
        *self.last.borrow_mut() = Instant::now();
        self.observer.emit(&format!(
            "index: {}: processed={processed}/{total} unprocessed={} committed={saved}",
            self.name,
            total.saturating_sub(processed)
        ));
    }

    pub(crate) fn finish(mut self, state: &'static str) {
        let seconds = self.start.elapsed().as_secs_f64();
        self.observer.add_seconds(self.name, seconds);
        self.finished = true;
        self.observer
            .emit(&format!("index: {}: {state} ({seconds:.6}s)", self.name));
    }
}

impl Drop for Stage<'_> {
    fn drop(&mut self) {
        if !self.finished {
            let seconds = self.start.elapsed().as_secs_f64();
            self.observer.add_seconds(self.name, seconds);
            self.observer.emit(&format!("index: {}: unfinished; only earlier commit notifications confirm saves ({seconds:.6}s)", self.name));
        }
    }
}

impl Drop for Observer {
    fn drop(&mut self) {
        self.emit(&format!("index: observations: {}", self.snapshot()));
    }
}

#[cfg(test)]
pub(crate) fn recording() -> (Observer, Rc<RefCell<Vec<String>>>) {
    let lines = Rc::new(RefCell::new(Vec::new()));
    let output = Rc::clone(&lines);
    (
        Observer::with_reporter(move |line| output.borrow_mut().push(line.to_owned())),
        lines,
    )
}

#[cfg(test)]
mod tests;
