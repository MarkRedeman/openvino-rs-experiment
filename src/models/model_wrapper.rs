use crate::domain::error::InferenceError;
use crate::domain::model::InferenceModel;
use crate::domain::types::{InferenceContext, InferenceInput, InferenceOutput, ModelInfo};
use crate::models::act::ActModel;
use crate::models::vision::VisionModel;

pub enum ModelWrapper {
    Vision(VisionModel),
    Act(ActModel),
}

impl InferenceModel for ModelWrapper {
    fn info(&self) -> &ModelInfo {
        match self {
            Self::Vision(model) => model.info(),
            Self::Act(model) => model.info(),
        }
    }

    fn infer<'a>(
        &mut self,
        input: InferenceInput<'a>,
        ctx: &InferenceContext,
    ) -> Result<InferenceOutput, InferenceError> {
        match self {
            Self::Vision(model) => model.infer(input, ctx),
            Self::Act(model) => model.infer(input, ctx),
        }
    }
}

impl ModelWrapper {
    pub fn as_vision_mut(&mut self) -> Option<&mut VisionModel> {
        match self {
            Self::Vision(model) => Some(model),
            _ => None,
        }
    }

    pub fn as_act_mut(&mut self) -> Option<&mut ActModel> {
        match self {
            Self::Act(model) => Some(model),
            _ => None,
        }
    }
}
