use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum InferenceError {
    InvalidInput(String),
    Runtime(anyhow::Error),
}

impl Display for InferenceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid inference input: {msg}"),
            Self::Runtime(err) => write!(f, "inference runtime error: {err}"),
        }
    }
}

impl Error for InferenceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidInput(_) => None,
            Self::Runtime(err) => Some(err.as_ref()),
        }
    }
}

impl From<anyhow::Error> for InferenceError {
    fn from(value: anyhow::Error) -> Self {
        Self::Runtime(value)
    }
}
