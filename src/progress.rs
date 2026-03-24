use std::io::{IsTerminal, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

const FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const TICK_MS: u64 = 80;

struct State {
    message: String,
    done: bool,
}

pub struct Spinner {
    state: Arc<Mutex<State>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Spinner {
    pub fn new(msg: &str) -> Self {
        let state = Arc::new(Mutex::new(State {
            message: msg.to_string(),
            done: false,
        }));

        let thread = if std::io::stderr().is_terminal() {
            let state = Arc::clone(&state);
            Some(thread::spawn(move || {
                let mut err = std::io::stderr();
                let mut i = 0;
                loop {
                    let s = state.lock().unwrap_or_else(|e| e.into_inner());
                    if s.done {
                        break;
                    }
                    eprint!("\r\x1b[2K{} {}", FRAMES[i % FRAMES.len()], s.message);
                    let _ = err.flush();
                    drop(s);
                    thread::sleep(Duration::from_millis(TICK_MS));
                    i += 1;
                }
            }))
        } else {
            None
        };

        Self { state, thread }
    }

    pub fn set_message(&self, msg: &str) {
        if let Ok(mut s) = self.state.lock() {
            s.message = msg.to_string();
        }
    }

    pub fn finish(self, msg: &str) {
        {
            let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
            s.done = true;
        }
        if let Some(t) = self.thread {
            let _ = t.join();
            eprint!("\r\x1b[2K");
            let _ = std::io::stderr().flush();
        }
        done(msg);
    }
}

/// Print a ✓ completion line without a spinner.
pub fn done(msg: &str) {
    eprintln!("\x1b[32m✓\x1b[0m {msg}");
}
