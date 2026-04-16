use crate::KbError;

/// Strip markdown code fences from LLM output.
pub fn strip_code_fences(s: &str) -> String {
    let trimmed = s.trim();
    if let Some(rest) = trimmed.strip_prefix("```json") {
        rest.trim_start_matches('\n')
            .trim_end_matches("```")
            .trim()
            .to_string()
    } else if let Some(rest) = trimmed.strip_prefix("```") {
        rest.trim_start_matches('\n')
            .trim_end_matches("```")
            .trim()
            .to_string()
    } else {
        trimmed.to_string()
    }
}

/// Extract and deserialize the first JSON object from LLM output.
///
/// Tolerates preamble text, code fences, and trailing commentary.
pub fn extract_json<T: serde::de::DeserializeOwned>(raw: &str) -> Result<T, KbError> {
    let stripped = strip_code_fences(raw);
    let start = stripped.find('{');
    let end = stripped.rfind('}');
    let slice = match (start, end) {
        (Some(s), Some(e)) if e >= s => &stripped[s..=e],
        _ => {
            return Err(KbError::Other(format!(
                "no JSON object in response: {raw:?}"
            )))
        }
    };
    serde_json::from_str(slice).map_err(|e| KbError::Other(format!("json parse: {e} in {slice:?}")))
}
