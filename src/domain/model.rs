use crate::domain::error::InferenceError;
use crate::domain::types::{InferenceContext, InferenceInput, InferenceOutput, ModelInfo};

pub trait InferenceModel {
    fn info(&self) -> &ModelInfo;

    fn infer<'a>(
        &mut self,
        input: InferenceInput<'a>,
        _ctx: &InferenceContext,
    ) -> Result<InferenceOutput, InferenceError>;
}
