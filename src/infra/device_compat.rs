//! Device compatibility knowledge base and compile-error diagnostics.
//!
//! This module provides:
//! - Known per-device operation and precision limitations.
//! - Error-pattern matching against OpenVINO compile error detail strings.
//! - Structured [`Diagnosis`] generation with human-readable causes and
//!   actionable suggestions.

use crate::infra::model_info::ModelInfo;
use std::fmt;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Structured diagnosis produced when a model fails to compile for a device.
#[derive(Debug, Clone)]
pub struct Diagnosis {
    /// One-line summary of the failure.
    pub summary: String,
    /// Likely causes, in decreasing order of probability.
    pub likely_causes: Vec<String>,
    /// Actionable suggestions for the user.
    pub suggestions: Vec<String>,
}

impl fmt::Display for Diagnosis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.summary)?;
        if !self.likely_causes.is_empty() {
            writeln!(f, "  Likely causes:")?;
            for cause in &self.likely_causes {
                writeln!(f, "    - {cause}")?;
            }
        }
        if !self.suggestions.is_empty() {
            writeln!(f, "  Suggestions:")?;
            for suggestion in &self.suggestions {
                writeln!(f, "    - {suggestion}")?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Known device limitations
// ---------------------------------------------------------------------------

/// Operations known to be unsupported or problematic on the NPU.
const NPU_UNSUPPORTED_OPS: &[&str] = &[
    "NonMaxSuppression",
    "TopK",
    "NonZero",
    "Loop",
    "If",
    "TensorIterator",
    "DetectionOutput",
    "ExperimentalDetectronDetectionOutput",
    "ExperimentalDetectronGenerateProposalsSingleImage",
    "ExperimentalDetectronROIFeatureExtractor",
    "ExperimentalDetectronTopKROIs",
    "ROIPooling",
    "ROIAlign",
    "DeformableConvolution",
    "CTCGreedyDecoder",
    "CTCGreedyDecoderSeqLen",
    "Bucketize",
    "AdaptiveAvgPool",
    "AdaptiveMaxPool",
];

/// Operations that may cause issues on the NPU due to dynamic shapes or
/// integer types that the NPU handles poorly.
const NPU_PROBLEMATIC_OPS: &[&str] = &[
    "ShapeOf",
    "Range",
    "Select",
    "ReduceMax",
    "ReduceMin",
    "ReduceProd",
    "Unique",
    "ScatterNDUpdate",
    "ScatterElementsUpdate",
    "GatherND",
];

/// Operations known to be problematic on the Intel GPU plugin.
const GPU_PROBLEMATIC_OPS: &[&str] = &[
    "CumSum",
    "Unique",
    "NonZero",
    "Bucketize",
    "AdaptiveAvgPool",
    "AdaptiveMaxPool",
    "CTCGreedyDecoder",
    "CTCGreedyDecoderSeqLen",
];

/// Supported precisions by device.
pub fn supported_precisions(device: &str) -> &'static [&'static str] {
    let upper = device.to_uppercase();
    let base = upper.split(':').next().unwrap_or(&upper);
    match base {
        "NPU" => &["FP16", "INT8"],
        "GPU" => &["FP32", "FP16", "INT8"],
        "CPU" => &["FP32", "FP16", "BF16", "INT8"],
        _ => &[],
    }
}

// ---------------------------------------------------------------------------
// Model-level compatibility checks (static analysis)
// ---------------------------------------------------------------------------

/// Statically analyse a model against a device and return potential issues.
///
/// This does *not* attempt to compile the model.  It only checks the model's
/// operation types and element types against known device limitations.
pub fn check_static_compat(device: &str, info: &ModelInfo) -> Vec<String> {
    let mut issues = Vec::new();
    let upper = device.to_uppercase();
    let base = upper.split(':').next().unwrap_or(&upper);

    // Check for unsupported ops.
    let unsupported_list: &[&str] = match base {
        "NPU" => NPU_UNSUPPORTED_OPS,
        _ => &[],
    };
    for op in unsupported_list {
        if let Some(&count) = info.op_counts.get(*op) {
            issues.push(format!("{count}x {op} — not supported on {base}"));
        }
    }

    // Check for problematic ops.
    let problematic_list: &[&str] = match base {
        "NPU" => NPU_PROBLEMATIC_OPS,
        "GPU" => GPU_PROBLEMATIC_OPS,
        _ => &[],
    };
    for op in problematic_list {
        if let Some(&count) = info.op_counts.get(*op) {
            issues.push(format!("{count}x {op} — may cause issues on {base}"));
        }
    }

    // Check for precision mismatches.
    let precs = supported_precisions(device);
    if !precs.is_empty() {
        for (elem_type, &count) in &info.element_type_counts {
            if !precs.contains(&elem_type.as_str()) {
                issues.push(format!(
                    "{count}x {elem_type} precision — {base} only supports {}",
                    precs.join(", ")
                ));
            }
        }
    }

    issues
}

// ---------------------------------------------------------------------------
// Error-pattern diagnostics (post-compile-failure)
// ---------------------------------------------------------------------------

/// Known error patterns and their human-readable descriptions.
struct ErrorPattern {
    /// Substring to search for in the error detail.
    needle: &'static str,
    /// Human-readable cause description.
    cause: &'static str,
    /// Suggestion for the user.
    suggestion: &'static str,
}

const ERROR_PATTERNS: &[ErrorPattern] = &[
    ErrorPattern {
        needle: "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE",
        cause: "The NPU does not support one or more operations or data types in this model",
        suggestion: "Convert the model to FP16, or use --device 'HETERO:NPU,CPU' to run unsupported ops on CPU",
    },
    ErrorPattern {
        needle: "ZE_RESULT_ERROR_UNSUPPORTED_VERSION",
        cause: "The NPU driver is too old for this version of OpenVINO",
        suggestion: "Upgrade the intel-npu-driver package, or run inside the Docker container which bundles matching drivers",
    },
    ErrorPattern {
        needle: "could not execute a primitive",
        cause: "The GPU failed to compile a low-level compute kernel for this model",
        suggestion: "Try --device 'HETERO:GPU,CPU', convert the model to FP16, or use CPU/NPU instead",
    },
    ErrorPattern {
        needle: "ProgramBuilder build failed",
        cause: "The GPU program builder could not construct the execution graph",
        suggestion: "This model likely uses operations or tensor patterns unsupported by the Intel GPU plugin. Try --device 'HETERO:GPU,CPU' or use CPU instead",
    },
    ErrorPattern {
        needle: "Both VCL and MLIR compiler load failed",
        cause: "The NPU plugin compiler libraries are not installed",
        suggestion: "Install the intel-npu-compiler package, or use the Docker container which bundles the compiler",
    },
    ErrorPattern {
        needle: "Unable to find the `openvino_c` library",
        cause: "The OpenVINO shared libraries are not in the library search path",
        suggestion: "Use the run-inference.sh launcher script, set LD_LIBRARY_PATH to include the lib/ directory, or run inside the Docker container",
    },
];

/// Analyse a compile-model error and produce a structured [`Diagnosis`].
///
/// `device` is the device string that was used (e.g. `"NPU"`, `"GPU"`,
/// `"HETERO:NPU,CPU"`).  `error_detail` is the string returned by
/// [`super::diagnostics::get_last_openvino_error()`].  `model_info` is an
/// optional pre-parsed [`ModelInfo`] for the model that failed to compile.
pub fn diagnose_compile_error(
    device: &str,
    error_detail: &str,
    model_info: Option<&ModelInfo>,
) -> Diagnosis {
    let mut causes = Vec::new();
    let mut suggestions = Vec::new();

    // Match known error patterns.
    for pattern in ERROR_PATTERNS {
        if error_detail.contains(pattern.needle) {
            causes.push(pattern.cause.to_string());
            suggestions.push(pattern.suggestion.to_string());
        }
    }

    // If we have model info, add static compatibility analysis.
    if let Some(info) = model_info {
        let issues = check_static_compat(device, info);
        for issue in issues {
            causes.push(issue);
        }
    }

    // Deduplicate suggestions.
    suggestions.dedup();

    // Fall back if nothing matched.
    if causes.is_empty() {
        causes.push(format!(
            "The model could not be compiled for device '{device}' (unknown reason)"
        ));
        suggestions.push(
            "Try a different device (e.g. CPU) or check the OpenVINO documentation".to_string(),
        );
    }

    let summary = format!("Model compilation failed for device '{device}'");

    Diagnosis {
        summary,
        likely_causes: causes,
        suggestions,
    }
}
