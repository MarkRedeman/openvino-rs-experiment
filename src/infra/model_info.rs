//! Parse OpenVINO IR model XML to extract structural information.
//!
//! This module reads the `<layers>` section of an OpenVINO IR (`.xml`) file and
//! produces a [`ModelInfo`] summary that includes input/output port descriptions,
//! per-operation-type counts, and per-element-type counts.  The information is
//! used by the `check` command and by compile-error diagnostics to explain *why*
//! a model may be incompatible with a given device.

use anyhow::{Context, Result};
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use std::collections::BTreeMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Summary of a model's structure, extracted from the IR XML.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// The `name` attribute on the top-level `<net>` element.
    pub name: String,
    /// Model inputs (`Parameter` layers).
    pub inputs: Vec<PortInfo>,
    /// Model outputs (`Result` layers).
    pub outputs: Vec<PortInfo>,
    /// Operation type → occurrence count (excludes `Parameter`, `Const`, `Result`).
    pub op_counts: BTreeMap<String, usize>,
    /// Element-type (precision) tag → occurrence count across all output ports.
    pub element_type_counts: BTreeMap<String, usize>,
    /// Total number of layers in the graph (including `Parameter`/`Const`/`Result`).
    pub total_layers: usize,
}

/// Description of a single model input or output port.
#[derive(Debug, Clone)]
pub struct PortInfo {
    /// Human-readable name (from the layer `name` or port `names` attribute).
    pub name: String,
    /// Dimension list, e.g. `["1", "3", "800", "992"]`.  May contain `"?"` for
    /// dynamic dimensions.
    pub shape: Vec<String>,
    /// Element type string, e.g. `"f32"`, `"f16"`, `"i64"`.
    pub element_type: String,
}

impl std::fmt::Display for ModelInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Model: {}", self.name)?;

        // Inputs.
        for port in &self.inputs {
            writeln!(
                f,
                "  Input : {} [{}] {}",
                port.name,
                port.shape_str(),
                port.element_type
            )?;
        }

        // Outputs.
        for port in &self.outputs {
            writeln!(
                f,
                "  Output: {} [{}] {}",
                port.name,
                port.shape_str(),
                port.element_type
            )?;
        }

        // Op summary.
        let unique = self.op_counts.len();
        let compute_ops: usize = self.op_counts.values().sum();
        writeln!(
            f,
            "  Ops   : {unique} unique types, {compute_ops} compute layers ({} total)",
            self.total_layers
        )?;

        // Element types.
        let types: Vec<String> = self
            .element_type_counts
            .iter()
            .map(|(t, c)| format!("{t} ({c})"))
            .collect();
        writeln!(f, "  Types : {}", types.join(", "))?;

        Ok(())
    }
}

impl PortInfo {
    /// Format the shape as a comma-separated string, e.g. `"1,3,800,992"`.
    pub fn shape_str(&self) -> String {
        if self.shape.is_empty() {
            "scalar".to_string()
        } else {
            self.shape.join(",")
        }
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Parse an OpenVINO IR `.xml` file and return a [`ModelInfo`].
pub fn parse_model_xml(path: &Path) -> Result<ModelInfo> {
    let xml_bytes = std::fs::read(path)
        .with_context(|| format!("failed to read model XML: {}", path.display()))?;

    let mut reader = Reader::from_reader(xml_bytes.as_slice());
    reader.config_mut().trim_text(true);

    let mut model_name = String::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut op_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut element_type_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut total_layers: usize = 0;

    // State tracked while inside a `<layer>` element.
    let mut current_layer_type: Option<String> = None;
    let mut current_layer_name: Option<String> = None;
    let mut current_layer_element_type: Option<String> = None;
    let mut current_layer_shape: Option<String> = None;
    let mut current_output_name: Option<String> = None;
    // Accumulate dims from `<dim>` children of the first `<port>` inside `<output>`.
    let mut current_dims: Vec<String> = Vec::new();
    let mut in_output_section = false;
    let mut captured_first_port = false;

    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Eof) => break,

            // ----- <net name="..."> -----
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"net" => {
                for attr in e.attributes().flatten() {
                    if attr.key.as_ref() == b"name" {
                        model_name = String::from_utf8_lossy(&attr.value).into_owned();
                    }
                }
            }

            // ----- <layer ...> -----
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"layer" => {
                total_layers += 1;
                let mut layer_type = String::new();
                let mut layer_name = String::new();
                let mut output_names = String::new();
                for attr in e.attributes().flatten() {
                    match attr.key.as_ref() {
                        b"type" => layer_type = String::from_utf8_lossy(&attr.value).into_owned(),
                        b"name" => layer_name = String::from_utf8_lossy(&attr.value).into_owned(),
                        b"output_names" => {
                            output_names = String::from_utf8_lossy(&attr.value).into_owned();
                        }
                        _ => {}
                    }
                }

                current_layer_type = Some(layer_type.clone());
                current_layer_name = if !output_names.is_empty() {
                    Some(output_names)
                } else {
                    Some(layer_name)
                };
                current_layer_element_type = None;
                current_layer_shape = None;
                current_dims.clear();
                in_output_section = false;
                captured_first_port = false;
                current_output_name = None;

                // Count non-structural ops.
                match layer_type.as_str() {
                    "Parameter" | "Const" | "Result" => {}
                    _ => {
                        *op_counts.entry(layer_type).or_insert(0) += 1;
                    }
                }
            }

            // ----- <data element_type="..." shape="..."/> -----
            Ok(Event::Empty(ref e)) if e.name().as_ref() == b"data" => {
                if current_layer_type.is_some() {
                    for attr in e.attributes().flatten() {
                        match attr.key.as_ref() {
                            b"element_type" => {
                                current_layer_element_type =
                                    Some(String::from_utf8_lossy(&attr.value).into_owned());
                            }
                            b"shape" => {
                                current_layer_shape =
                                    Some(String::from_utf8_lossy(&attr.value).into_owned());
                            }
                            _ => {}
                        }
                    }
                }
            }

            // ----- <output> -----
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"output" => {
                in_output_section = true;
                captured_first_port = false;
            }

            // ----- </output> -----
            Ok(Event::End(ref e)) if e.name().as_ref() == b"output" => {
                in_output_section = false;
            }

            // ----- <port precision="..." names="..."> -----
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"port" && in_output_section => {
                if !captured_first_port {
                    for attr in e.attributes().flatten() {
                        match attr.key.as_ref() {
                            b"precision" => {
                                let prec = String::from_utf8_lossy(&attr.value).into_owned();
                                *element_type_counts.entry(prec).or_insert(0) += 1;
                            }
                            b"names" => {
                                current_output_name =
                                    Some(String::from_utf8_lossy(&attr.value).into_owned());
                            }
                            _ => {}
                        }
                    }
                    current_dims.clear();
                }
            }

            // ----- </port> -----
            Ok(Event::End(ref e)) if e.name().as_ref() == b"port" && in_output_section => {
                if !captured_first_port {
                    captured_first_port = true;
                }
            }

            // ----- <dim>N</dim> -----
            Ok(Event::Start(ref e))
                if e.name().as_ref() == b"dim" && in_output_section && !captured_first_port => {}

            Ok(Event::Text(ref e)) if in_output_section && !captured_first_port => {
                let text = e.unescape().unwrap_or_default().trim().to_string();
                if !text.is_empty() {
                    current_dims.push(text);
                }
            }

            // ----- </layer> -----
            Ok(Event::End(ref e)) if e.name().as_ref() == b"layer" => {
                if let Some(ref layer_type) = current_layer_type {
                    match layer_type.as_str() {
                        "Parameter" => {
                            let name = current_output_name
                                .take()
                                .or_else(|| current_layer_name.take())
                                .unwrap_or_default();
                            let element_type = current_layer_element_type
                                .take()
                                .unwrap_or_else(|| "unknown".into());
                            let shape = if current_dims.is_empty() {
                                // Fall back to parsing the shape attribute from <data>.
                                parse_shape_attr(current_layer_shape.as_deref())
                            } else {
                                current_dims.clone()
                            };
                            inputs.push(PortInfo {
                                name,
                                shape,
                                element_type,
                            });
                        }
                        "Result" => {
                            let name = current_layer_name.take().unwrap_or_default();
                            // Result layers usually don't carry their own shape/type;
                            // strip the `/sink_port_0` suffix if present.
                            let name = name
                                .strip_suffix("/sink_port_0")
                                .unwrap_or(&name)
                                .to_string();
                            let element_type = current_layer_element_type
                                .take()
                                .unwrap_or_else(|| "unknown".into());
                            let shape = if current_dims.is_empty() {
                                parse_shape_attr(current_layer_shape.as_deref())
                            } else {
                                current_dims.clone()
                            };
                            outputs.push(PortInfo {
                                name,
                                shape,
                                element_type,
                            });
                        }
                        _ => {}
                    }
                }
                current_layer_type = None;
                current_layer_name = None;
                current_layer_element_type = None;
                current_layer_shape = None;
                current_dims.clear();
                in_output_section = false;
                captured_first_port = false;
                current_output_name = None;
            }

            Ok(_) => {}
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "XML parse error at position {}: {e}",
                    reader.error_position()
                ));
            }
        }
        buf.clear();
    }

    Ok(ModelInfo {
        name: model_name,
        inputs,
        outputs,
        op_counts,
        element_type_counts,
        total_layers,
    })
}

/// Parse a comma-separated shape string from the `<data shape="...">` attribute,
/// e.g. `"1, 3, 800, 992"` → `["1", "3", "800", "992"]`.
fn parse_shape_attr(shape: Option<&str>) -> Vec<String> {
    match shape {
        Some(s) if !s.is_empty() => s.split(',').map(|d| d.trim().to_string()).collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn parse_fish_detection_model() {
        let path = PathBuf::from("models/fish-detection/model.xml");
        if !path.exists() {
            eprintln!("skipping test — model file not found");
            return;
        }
        let info = parse_model_xml(&path).unwrap();
        assert_eq!(info.name, "main_graph");
        assert!(!info.inputs.is_empty());
        assert!(!info.outputs.is_empty());
        assert!(info.total_layers > 0);
        assert!(!info.op_counts.is_empty());
        println!("{info}");
    }
}
