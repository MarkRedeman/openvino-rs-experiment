use crate::act::{ActInputTensors, ActOutput};
use crate::engine::OutputBuffer;
use openvino::Tensor;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Classification,
    Detection,
    Act,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_type: ModelType,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
}

pub enum InferenceInput<'a> {
    Image(&'a Tensor),
    Act(&'a ActInputTensors),
}

pub enum InferenceOutput {
    Tensor(Vec<f32>),
    MultiTensor(HashMap<String, OutputBuffer>),
    Act(ActOutput),
}

#[derive(Debug, Clone, Default)]
pub struct InferenceContext {
    pub labels: Vec<String>,
}
