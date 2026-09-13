//! Normalizes real GGUF (llama.cpp convention) Qwen2 tensor names (e.g.
//! `blk.0.attn_q.weight`, `token_embd.weight`) into the exact same
//! canonical Model Artifact names `loaders/huggingface`'s own
//! `naming::normalize_tensor_name` produces from Hugging Face Safetensors
//! names -- `magnetar-runtime`'s Qwen Component resolves weight edges by
//! this canonical identity alone and has no notion of which bundle format
//! (Hugging Face bundle or GGUF file) a checkpoint originally came from.
//! GGUF's own tensor-naming convention (`blk.N.*`, `token_embd`,
//! `output_norm`, `output`) is documented by `ggml-org/llama.cpp`'s GGUF
//! writer/reader and is stable across llama.cpp-produced GGUF files for
//! the Qwen2/Qwen2.5 architecture family.

use magnetar_runtime::production_model_ingestion::ProductionIngestionError;

/// Normalizes one real GGUF Qwen2 tensor name into its canonical Model
/// Artifact name, or rejects it structurally. Mirrors `loaders/huggingface`'s
/// `normalize_tensor_name` exactly in its accepted/rejected shapes:
/// - `blk.N.attn_{q,k,v}.bias` is accepted and canonicalized to
///   `layers.N.self_attn.{q,k,v}_bias` (Qwen2/2.5's real architectural
///   QKV bias, the only bias GGUF Qwen2 exports carry).
/// - any other bias tensor is rejected with `UnsupportedFormat`: the
///   production Qwen Component's graph does not add a bias term to any
///   other projection, so silently dropping one would produce numerically
///   wrong results rather than a fail-closed error.
/// - an unrecognized name is rejected with `MalformedMetadata` naming it.
pub fn normalize_tensor_name(raw: &str) -> Result<String, ProductionIngestionError> {
    if let Some(suffix) = raw.strip_suffix(".bias") {
        if let Some(layer_suffix) = suffix.strip_prefix("blk.")
            && let Some((layer_index, rest)) = layer_suffix.split_once('.')
            && layer_index.parse::<u64>().is_ok()
        {
            let canonical_bias_suffix = match rest {
                "attn_q" => Some("self_attn.q_bias"),
                "attn_k" => Some("self_attn.k_bias"),
                "attn_v" => Some("self_attn.v_bias"),
                _ => None,
            };
            if let Some(canonical_bias_suffix) = canonical_bias_suffix {
                return Ok(format!("layers.{layer_index}.{canonical_bias_suffix}"));
            }
        }
        return Err(ProductionIngestionError::UnsupportedFormat {
            reason: format!(
                "tensor '{raw}' is a bias term for '{suffix}'; the production Qwen Component \
                 graph only supports attention q/k/v bias terms"
            ),
        });
    }
    let Some(name) = raw.strip_suffix(".weight") else {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: format!("tensor '{raw}' does not end in '.weight' or '.bias'"),
        });
    };

    if name == "token_embd" {
        return Ok("token_embedding".to_string());
    }
    if name == "output_norm" {
        return Ok("final_norm".to_string());
    }
    if name == "output" {
        return Ok("lm_head".to_string());
    }
    if let Some(layer_suffix) = name.strip_prefix("blk.") {
        let (layer_index, rest) = layer_suffix.split_once('.').ok_or_else(|| {
            ProductionIngestionError::MalformedMetadata {
                reason: format!("tensor '{raw}' has an unrecognized per-layer name shape"),
            }
        })?;
        layer_index
            .parse::<u64>()
            .map_err(|_| ProductionIngestionError::MalformedMetadata {
                reason: format!("tensor '{raw}' has a non-numeric layer index '{layer_index}'"),
            })?;
        let canonical_suffix = match rest {
            "attn_norm" => "input_norm",
            "ffn_norm" => "post_attn_norm",
            "attn_q" => "self_attn.q_proj",
            "attn_k" => "self_attn.k_proj",
            "attn_v" => "self_attn.v_proj",
            "attn_output" => "self_attn.o_proj",
            "ffn_gate" => "mlp.gate_proj",
            "ffn_up" => "mlp.up_proj",
            "ffn_down" => "mlp.down_proj",
            other => {
                return Err(ProductionIngestionError::MalformedMetadata {
                    reason: format!(
                        "tensor '{raw}' names an unrecognized per-layer field '{other}'"
                    ),
                });
            }
        };
        return Ok(format!("layers.{layer_index}.{canonical_suffix}"));
    }

    Err(ProductionIngestionError::MalformedMetadata {
        reason: format!("tensor '{raw}' does not match any recognized Qwen2 GGUF tensor name shape"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_top_level_tensors() {
        assert_eq!(
            normalize_tensor_name("token_embd.weight").unwrap(),
            "token_embedding"
        );
        assert_eq!(
            normalize_tensor_name("output_norm.weight").unwrap(),
            "final_norm"
        );
        assert_eq!(normalize_tensor_name("output.weight").unwrap(), "lm_head");
    }

    #[test]
    fn normalizes_per_layer_tensors() {
        assert_eq!(
            normalize_tensor_name("blk.0.attn_q.weight").unwrap(),
            "layers.0.self_attn.q_proj"
        );
        assert_eq!(
            normalize_tensor_name("blk.12.ffn_down.weight").unwrap(),
            "layers.12.mlp.down_proj"
        );
        assert_eq!(
            normalize_tensor_name("blk.3.attn_norm.weight").unwrap(),
            "layers.3.input_norm"
        );
        assert_eq!(
            normalize_tensor_name("blk.3.ffn_norm.weight").unwrap(),
            "layers.3.post_attn_norm"
        );
    }

    #[test]
    fn normalizes_qkv_bias_tensors() {
        assert_eq!(
            normalize_tensor_name("blk.0.attn_q.bias").unwrap(),
            "layers.0.self_attn.q_bias"
        );
        assert_eq!(
            normalize_tensor_name("blk.0.attn_k.bias").unwrap(),
            "layers.0.self_attn.k_bias"
        );
        assert_eq!(
            normalize_tensor_name("blk.7.attn_v.bias").unwrap(),
            "layers.7.self_attn.v_bias"
        );
    }

    #[test]
    fn rejects_bias_tensors_outside_qkv() {
        let error = normalize_tensor_name("blk.0.attn_output.bias").unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::UnsupportedFormat { .. }
        ));
        let error = normalize_tensor_name("blk.0.ffn_gate.bias").unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::UnsupportedFormat { .. }
        ));
        let error = normalize_tensor_name("output.bias").unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::UnsupportedFormat { .. }
        ));
    }

    #[test]
    fn rejects_unrecognized_names() {
        let error = normalize_tensor_name("blk.0.some_new_field.weight").unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::MalformedMetadata { .. }
        ));
        let error2 = normalize_tensor_name("something_else_entirely").unwrap_err();
        assert!(matches!(
            error2,
            ProductionIngestionError::MalformedMetadata { .. }
        ));
    }
}
