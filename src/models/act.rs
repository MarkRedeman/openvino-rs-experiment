use anyhow::Result;
use std::path::Path;

use crate::act::{ActEngine, ActMetadata, read_act_output};
use crate::domain::error::InferenceError;
use crate::domain::model::InferenceModel;
use crate::domain::types::{
    InferenceContext, InferenceInput, InferenceOutput, ModelInfo, ModelType,
};

pub struct ActModel {
    engine: ActEngine,
    metadata: ActMetadata,
    info: ModelInfo,
}

impl ActModel {
    pub fn new(
        model_xml: &Path,
        model_bin: &Path,
        device: &str,
        metadata: ActMetadata,
    ) -> Result<Self> {
        let engine = ActEngine::new(model_xml, model_bin, device)?;
        let info = ModelInfo {
            model_type: ModelType::Act,
            input_names: vec![
                "state".to_string(),
                "images.gripper".to_string(),
                "images.overview".to_string(),
            ],
            output_names: vec!["action".to_string()],
        };

        Ok(Self {
            engine,
            metadata,
            info,
        })
    }

    pub fn metadata(&self) -> &ActMetadata {
        &self.metadata
    }

    pub fn engine_mut(&mut self) -> &mut ActEngine {
        &mut self.engine
    }
}

impl InferenceModel for ActModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn infer<'a>(
        &mut self,
        input: InferenceInput<'a>,
        _ctx: &InferenceContext,
    ) -> Result<InferenceOutput, InferenceError> {
        let tensors = match input {
            InferenceInput::Act(tensors) => tensors,
            InferenceInput::Image(_) => {
                return Err(InferenceError::InvalidInput(
                    "ACT model requires ACT input tensors".to_string(),
                ));
            }
        };

        let mut request = self.engine.create_request()?;
        self.engine.run_request(&mut request, tensors)?;
        let output = read_act_output(&request, &self.metadata)?;

        Ok(InferenceOutput::Act(output))
    }
}
