#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionType {
    Interactive,
    Automated,
}

impl SessionType {
    pub fn as_str(self) -> &'static str {
        match self {
            SessionType::Interactive => "interactive",
            SessionType::Automated => "automated",
        }
    }
}

pub fn classify_session(first_turn: &str, automated_origin: bool) -> SessionType {
    if automated_origin {
        SessionType::Automated
    } else {
        classify_first_turn(first_turn)
    }
}

const AUTOMATED_MARKERS: &[&str] = &[
    "<command-message>",
    "<user_action>",
    "<local-command-stdout>",
    "<local-command-caveat>",
    "<bash-input>",
    "[Request interrupted by user",
    "This session is being continued",
];

pub fn classify_first_turn(first_turn: &str) -> SessionType {
    let head = first_turn.trim_start();
    if AUTOMATED_MARKERS
        .iter()
        .any(|marker| head.starts_with(marker))
    {
        SessionType::Automated
    } else {
        SessionType::Interactive
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod integration_tests;

#[cfg(test)]
mod reclassification_tests;

#[cfg(test)]
mod benchmark;
