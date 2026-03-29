use anyhow::Result;
use openvino::Tensor;
use std::path::Path;

use crate::domain::error::InferenceError;
use crate::domain::model::InferenceModel;
use crate::domain::types::{
    InferenceContext, InferenceInput, InferenceOutput, ModelInfo, ModelType,
};
use crate::engine::Engine;

pub struct VisionModel {
    engine: Engine,
    info: ModelInfo,
    multi_output: bool,
}

impl VisionModel {
    pub fn new(
        model_xml: &Path,
        model_bin: &Path,
        device: &str,
        reference_tensor: &Tensor,
        model_type: ModelType,
    ) -> Result<Self> {
        let engine = Engine::new(model_xml, model_bin, device, reference_tensor)?;
        let output_names = engine.output_names().to_vec();
        let input_names = vec![engine.input_name().to_string()];
        let multi_output = output_names.len() > 1;

        Ok(Self {
            engine,
            info: ModelInfo {
                model_type,
                input_names,
                output_names,
            },
            multi_output,
        })
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }
}

impl InferenceModel for VisionModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn infer<'a>(
        &mut self,
        input: InferenceInput<'a>,
        _ctx: &InferenceContext,
    ) -> Result<InferenceOutput, InferenceError> {
        let tensor = match input {
            InferenceInput::Image(tensor) => tensor,
            InferenceInput::Act(_) => {
                return Err(InferenceError::InvalidInput(
                    "vision model requires image tensor input".to_string(),
                ));
            }
        };

        if self.multi_output {
            let outputs = self.engine.infer_multi(tensor)?;
            Ok(InferenceOutput::MultiTensor(outputs))
        } else {
            let output = self.engine.infer(tensor)?;
            Ok(InferenceOutput::Tensor(output))
        }
    }
}
