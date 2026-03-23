use anyhow::{Context, Result};
use std::path::Path;

/// Extract class label names from an OpenVINO IR model XML file.
///
/// Intel Geti / OTX exported models embed label metadata in the
/// `<rt_info><model_info>` section:
///
/// ```xml
/// <labels value="Clubs Spades Diamonds Hearts" />
/// ```
///
/// Returns an ordered `Vec<String>` of label names (index 0 = class 0, etc.),
/// or an empty vec if no label metadata is found.
pub fn parse_labels_from_model_xml(model_xml: &Path) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(model_xml)
        .with_context(|| format!("failed to read model XML: {}", model_xml.display()))?;

    // Look for the <labels value="..." /> element inside <model_info>.
    // We search for the pattern within <rt_info> to avoid matching layer names.
    let rt_info_start = match content.find("<rt_info>") {
        Some(pos) => pos,
        None => return Ok(Vec::new()),
    };

    let rt_info_section = &content[rt_info_start..];

    // Find <labels value="..."/>  (with or without spaces before />)
    let labels_tag = "<labels value=\"";
    let tag_start = match rt_info_section.find(labels_tag) {
        Some(pos) => pos + labels_tag.len(),
        None => return Ok(Vec::new()),
    };

    let remaining = &rt_info_section[tag_start..];
    let tag_end = match remaining.find('"') {
        Some(pos) => pos,
        None => return Ok(Vec::new()),
    };

    let labels_str = &remaining[..tag_end];
    if labels_str.is_empty() {
        return Ok(Vec::new());
    }

    let labels: Vec<String> = labels_str
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();

    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_parse_classification_labels() {
        let path = PathBuf::from("models/card-classification/model.xml");
        if path.exists() {
            let labels = parse_labels_from_model_xml(&path).unwrap();
            assert_eq!(labels, vec!["Clubs", "Spades", "Diamonds", "Hearts"]);
        }
    }

    #[test]
    fn test_parse_detection_labels() {
        let path = PathBuf::from("models/fish-detection/model.xml");
        if path.exists() {
            let labels = parse_labels_from_model_xml(&path).unwrap();
            assert_eq!(labels, vec!["Fish"]);
        }
    }
}
