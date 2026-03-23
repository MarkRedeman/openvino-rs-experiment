use anyhow::{Context, Result};
use openvino::{
    prepostprocess, CompiledModel, Core, DeviceType, ElementType, InferRequest, Layout, Model,
    ResizeAlgorithm, Tensor,
};
use std::collections::HashMap;
use std::path::Path;

/// Raw bytes read from an output tensor, together with its element type.
#[derive(Debug, Clone)]
pub struct OutputBuffer {
    pub bytes: Vec<u8>,
    pub element_type: ElementType,
}

impl OutputBuffer {
    /// Interpret the buffer as `&[f32]`.  Panics on type/alignment mismatch.
    pub fn as_f32(&self) -> &[f32] {
        assert!(
            self.bytes.len() % std::mem::size_of::<f32>() == 0,
            "buffer length not a multiple of f32"
        );
        unsafe {
            std::slice::from_raw_parts(
                self.bytes.as_ptr() as *const f32,
                self.bytes.len() / std::mem::size_of::<f32>(),
            )
        }
    }

    /// Interpret the buffer as `&[i64]`.  Panics on type/alignment mismatch.
    pub fn as_i64(&self) -> &[i64] {
        assert!(
            self.bytes.len() % std::mem::size_of::<i64>() == 0,
            "buffer length not a multiple of i64"
        );
        unsafe {
            std::slice::from_raw_parts(
                self.bytes.as_ptr() as *const i64,
                self.bytes.len() / std::mem::size_of::<i64>(),
            )
        }
    }
}

/// High-level wrapper around the OpenVINO inference lifecycle.
///
/// Holds a compiled model and provides `infer` / `infer_multi` methods.
pub struct Engine {
    compiled: CompiledModel,
    input_name: String,
    output_names: Vec<String>,
}

impl Engine {
    /// Create a new inference engine.
    ///
    /// The `reference_tensor` is a representative input tensor (e.g. the first
    /// image you want to classify). It is **not** consumed — it is only used to
    /// tell the OpenVINO pre-processing pipeline about the element type and
    /// shape of tensors you will feed at inference time.
    ///
    /// # Arguments
    ///
    /// * `model_xml`         – Path to the OpenVINO IR model definition (`.xml`).
    /// * `model_bin`         – Path to the OpenVINO IR weights (`.bin`).
    /// * `device`            – Target device string (`"CPU"`, `"GPU"`, `"AUTO"`, …).
    /// * `reference_tensor`  – A tensor whose shape/type matches future inputs.
    pub fn new(
        model_xml: &Path,
        model_bin: &Path,
        device: &str,
        reference_tensor: &Tensor,
    ) -> Result<Self> {
        let mut core = Core::new().context("failed to initialise OpenVINO core")?;

        let mut model = core
            .read_model_from_file(&model_xml.to_string_lossy(), &model_bin.to_string_lossy())
            .context("failed to read model from file")?;

        // Discover input name.
        let input_name = model
            .get_input_by_index(0)
            .and_then(|n| n.get_name())
            .context("failed to retrieve model input name")?;

        // Discover all output names.
        let mut output_names = Vec::new();
        for i in 0.. {
            match model.get_output_by_index(i).and_then(|n| n.get_name()) {
                Ok(name) => output_names.push(name),
                Err(_) => break,
            }
        }
        if output_names.is_empty() {
            anyhow::bail!("model has no outputs");
        }

        // Build a pre/post-process pipeline.
        let single_output = output_names.len() == 1;
        let new_model = Self::build_preprocess_pipeline(
            &mut model,
            &input_name,
            reference_tensor,
            single_output,
        )?;

        let device_type = Self::parse_device(device);
        let compiled = core
            .compile_model(&new_model, device_type)
            .context("failed to compile model")?;

        Ok(Self {
            compiled,
            input_name,
            output_names,
        })
    }

    /// Run synchronous inference and return the first output as `Vec<f32>`.
    ///
    /// This is a convenience wrapper for single-output models.
    pub fn infer(&mut self, input: &Tensor) -> Result<Vec<f32>> {
        let request = self.create_and_run(input)?;

        let output = request
            .get_tensor(&self.output_names[0])
            .context("failed to retrieve output tensor")?;

        let data = output
            .get_data::<f32>()
            .context("failed to read output tensor data")?;

        Ok(data.to_vec())
    }

    /// Run synchronous inference and return **all** outputs keyed by name.
    ///
    /// Each output is returned as raw bytes together with its element type so
    /// callers can cast to the appropriate Rust type.
    pub fn infer_multi(&mut self, input: &Tensor) -> Result<HashMap<String, OutputBuffer>> {
        let request = self.create_and_run(input)?;

        let mut map = HashMap::new();
        for name in &self.output_names {
            let tensor = request
                .get_tensor(name)
                .with_context(|| format!("failed to retrieve output tensor '{name}'"))?;

            let element_type = tensor
                .get_element_type()
                .context("failed to get element type")?;
            let raw = tensor
                .get_raw_data()
                .context("failed to get raw output data")?;

            map.insert(
                name.clone(),
                OutputBuffer {
                    bytes: raw.to_vec(),
                    element_type,
                },
            );
        }

        Ok(map)
    }

    /// The name assigned to the model's first input port.
    pub fn input_name(&self) -> &str {
        &self.input_name
    }

    /// The names of all output ports.
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn create_and_run(&mut self, input: &Tensor) -> Result<InferRequest> {
        let mut request: InferRequest = self
            .compiled
            .create_infer_request()
            .context("failed to create infer request")?;

        request
            .set_tensor(&self.input_name, input)
            .context("failed to set input tensor")?;

        request.infer().context("inference failed")?;

        Ok(request)
    }

    fn build_preprocess_pipeline(
        model: &mut Model,
        input_name: &str,
        reference_tensor: &Tensor,
        coerce_output_f32: bool,
    ) -> Result<Model> {
        let ppp = prepostprocess::Pipeline::new(model)
            .context("failed to create pre/post-process pipeline")?;

        // Input tensor info — derive element type and shape from the reference
        // tensor, then declare NHWC layout.
        let input_info = ppp
            .get_input_info_by_name(input_name)
            .context("failed to get input info")?;

        let mut tensor_info = input_info
            .get_tensor_info()
            .context("failed to get input tensor info")?;
        tensor_info
            .set_from(reference_tensor)
            .context("failed to set input tensor info from reference tensor")?;
        tensor_info
            .set_layout(Layout::new("NHWC").context("failed to create NHWC layout")?)
            .context("failed to set input layout")?;

        // Preprocessing steps — resize to whatever the model expects.
        let mut steps = input_info
            .get_steps()
            .context("failed to get preprocessing steps")?;
        steps
            .resize(ResizeAlgorithm::Linear)
            .context("failed to add resize step")?;

        // Model layout — most vision models use NCHW internally.
        let mut model_info = input_info
            .get_model_info()
            .context("failed to get model info")?;
        model_info
            .set_layout(Layout::new("NCHW").context("failed to create NCHW layout")?)
            .context("failed to set model layout")?;

        // For single-output models, coerce output to F32 for easy consumption.
        // For multi-output models (e.g. Geti detection: boxes=F32, labels=I64)
        // we leave outputs in their native types.
        if coerce_output_f32 {
            let output_info = ppp
                .get_output_info_by_index(0)
                .context("failed to get output info")?;
            let mut output_tensor_info = output_info
                .get_tensor_info()
                .context("failed to get output tensor info")?;
            output_tensor_info
                .set_element_type(ElementType::F32)
                .context("failed to set output element type")?;
        }

        ppp.build_new_model()
            .context("failed to build preprocessed model")
    }

    fn parse_device(device: &str) -> DeviceType<'static> {
        match device.to_uppercase().as_str() {
            "CPU" => DeviceType::CPU,
            "GPU" => DeviceType::GPU,
            _ => DeviceType::CPU,
        }
    }
}
