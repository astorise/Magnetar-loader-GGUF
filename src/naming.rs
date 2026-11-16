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
    if name == "rope_freqs" {
        // llama.cpp stores a per-frequency RoPE scaling (Llama 3.1+) as
        // precomputed factors, which no Model Component reads from GGUF.
        return Err(ProductionIngestionError::UnsupportedFormat {
            reason: "tensor 'rope_freqs.weight' declares a per-frequency RoPE scaling, which \
                     GGUF loading does not support"
                .into(),
        });
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
            // `execute-and-cache-moe-experts` F02 (llama.cpp's `qwen2moe`):
            // the router, the stacked routed experts, and the shared expert
            // with its scalar gate.
            "ffn_gate_inp" => "moe.router_proj",
            "ffn_gate_exps" => "moe.experts_gate_proj",
            "ffn_up_exps" => "moe.experts_up_proj",
            "ffn_down_exps" => "moe.experts_down_proj",
            "ffn_gate_shexp" => "moe.shared_gate_proj",
            "ffn_up_shexp" => "moe.shared_up_proj",
            "ffn_down_shexp" => "moe.shared_down_proj",
            "ffn_gate_inp_shexp" => "moe.shared_expert_gate",
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
        reason: format!(
            "tensor '{raw}' does not match any recognized Qwen2 GGUF tensor name shape"
        ),
    })
}

/// The canonical name of `raw` in a file of `architecture`: a `gemma4`
/// file's layered decoder names its norms differently from the others
/// (its `ffn_norm` precedes the MLP, its `post_attention_norm` follows
/// attention), carries per-head `q_norm`/`k_norm`, a per-layer scalar and
/// per-layer input weights, and a `rope_freqs` table the loader reads
/// itself. Tables packing every layer's per-layer inputs, its fused expert
/// gate and up projections and its router take internal names that
/// `layered_tensors.rs` replaces with the per-layer and stacked tensors.
pub fn normalize_tensor_name_for(
    raw: &str,
    architecture: &str,
) -> Result<String, ProductionIngestionError> {
    if architecture != "gemma4" {
        return normalize_tensor_name(raw);
    }
    // The router's input scale and the experts' output scales have no
    // `.weight` suffix.
    if let Some((layer_index, field)) = raw
        .strip_suffix(".scale")
        .and_then(|name| name.strip_prefix("blk."))
        .and_then(|layer| layer.split_once('.'))
        .filter(|(index, _)| index.parse::<u64>().is_ok())
    {
        let canonical = match field {
            "ffn_gate_inp" => crate::layered_tensors::ROUTER_SCALE,
            "ffn_down_exps" => "moe.per_expert_scale",
            other => {
                return Err(ProductionIngestionError::MalformedMetadata {
                    reason: format!("tensor '{raw}' scales an unrecognized field '{other}'"),
                });
            }
        };
        return Ok(format!("layers.{layer_index}.{canonical}"));
    }
    let name =
        raw.strip_suffix(".weight")
            .ok_or_else(|| ProductionIngestionError::MalformedMetadata {
                reason: format!("tensor '{raw}' does not end in '.weight' or '.scale'"),
            })?;
    match name {
        "token_embd" => return Ok("token_embedding".into()),
        "output_norm" => return Ok("final_norm".into()),
        "output" => return Ok("lm_head".into()),
        "rope_freqs" => return Ok("rope_freqs".into()),
        "per_layer_proj_norm" => return Ok("per_layer_projection_norm".into()),
        "per_layer_token_embd" => {
            return Ok(crate::layered_tensors::PER_LAYER_EMBEDDING_TABLE.into());
        }
        "per_layer_model_proj" => {
            return Ok(crate::layered_tensors::PER_LAYER_PROJECTION_TABLE.into());
        }
        _ => {}
    }
    let Some((layer_index, rest)) = name
        .strip_prefix("blk.")
        .and_then(|layer| layer.split_once('.'))
        .filter(|(index, _)| index.parse::<u64>().is_ok())
    else {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: format!("tensor '{raw}' does not match any recognized gemma4 tensor name"),
        });
    };
    let canonical_suffix = match rest {
        "attn_norm" => "input_norm",
        "post_attention_norm" => "post_attn_norm",
        "ffn_norm" => "pre_ffn_norm",
        "post_ffw_norm" => "post_ffn_norm",
        "pre_ffw_norm_2" => "pre_ffn_norm_2",
        "post_ffw_norm_1" => "post_ffn_norm_1",
        "post_ffw_norm_2" => "post_ffn_norm_2",
        "attn_q" => "self_attn.q_proj",
        "attn_k" => "self_attn.k_proj",
        "attn_v" => "self_attn.v_proj",
        "attn_output" => "self_attn.o_proj",
        "attn_q_norm" => "self_attn.q_norm",
        "attn_k_norm" => "self_attn.k_norm",
        "ffn_gate" => "mlp.gate_proj",
        "ffn_up" => "mlp.up_proj",
        "ffn_down" => "mlp.down_proj",
        "layer_output_scale" => "layer_scalar",
        "inp_gate" => "per_layer_input_gate",
        "proj" => "per_layer_projection",
        "post_norm" => "post_per_layer_input_norm",
        // Routed experts: the router and fused gate and up projections are
        // derived (`layered_tensors.rs`); separate gate, up and down
        // projections are already stacked.
        "ffn_gate_inp" => crate::layered_tensors::ROUTER_PROJECTION,
        "ffn_gate_up_exps" => crate::layered_tensors::EXPERTS_GATE_UP,
        "ffn_gate_exps" => "moe.experts_gate_proj",
        "ffn_up_exps" => "moe.experts_up_proj",
        "ffn_down_exps" => "moe.experts_down_proj",
        other => {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: format!(
                    "tensor '{raw}' names an unrecognized gemma4 per-layer field '{other}'"
                ),
            });
        }
    };
    Ok(format!("layers.{layer_index}.{canonical_suffix}"))
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
    fn normalizes_mixture_of_experts_tensors() {
        for (raw, canonical) in [
            ("blk.2.ffn_gate_inp.weight", "layers.2.moe.router_proj"),
            (
                "blk.2.ffn_gate_exps.weight",
                "layers.2.moe.experts_gate_proj",
            ),
            ("blk.2.ffn_up_exps.weight", "layers.2.moe.experts_up_proj"),
            (
                "blk.2.ffn_down_exps.weight",
                "layers.2.moe.experts_down_proj",
            ),
            (
                "blk.2.ffn_gate_shexp.weight",
                "layers.2.moe.shared_gate_proj",
            ),
            ("blk.2.ffn_up_shexp.weight", "layers.2.moe.shared_up_proj"),
            (
                "blk.2.ffn_down_shexp.weight",
                "layers.2.moe.shared_down_proj",
            ),
            (
                "blk.2.ffn_gate_inp_shexp.weight",
                "layers.2.moe.shared_expert_gate",
            ),
        ] {
            assert_eq!(normalize_tensor_name(raw).unwrap(), canonical);
        }
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

#[cfg(test)]
mod gemma4_tests {
    use super::*;

    /// The tensor names of llama.cpp's `gemma4` files (read from the
    /// headers of `google/gemma-4-*-it-qat-q4_0-gguf`).
    #[test]
    fn gemma4_tensors_take_the_layered_logical_names() {
        let cases = [
            ("token_embd.weight", "token_embedding"),
            ("output_norm.weight", "final_norm"),
            ("rope_freqs.weight", "rope_freqs"),
            ("per_layer_proj_norm.weight", "per_layer_projection_norm"),
            ("blk.0.attn_norm.weight", "layers.0.input_norm"),
            (
                "blk.0.post_attention_norm.weight",
                "layers.0.post_attn_norm",
            ),
            ("blk.0.ffn_norm.weight", "layers.0.pre_ffn_norm"),
            ("blk.0.post_ffw_norm.weight", "layers.0.post_ffn_norm"),
            ("blk.3.post_ffw_norm_2.weight", "layers.3.post_ffn_norm_2"),
            ("blk.4.attn_q_norm.weight", "layers.4.self_attn.q_norm"),
            ("blk.4.attn_k.weight", "layers.4.self_attn.k_proj"),
            ("blk.4.layer_output_scale.weight", "layers.4.layer_scalar"),
            ("blk.4.inp_gate.weight", "layers.4.per_layer_input_gate"),
            ("blk.4.proj.weight", "layers.4.per_layer_projection"),
            (
                "blk.4.post_norm.weight",
                "layers.4.post_per_layer_input_norm",
            ),
        ];
        for (raw, canonical) in cases {
            assert_eq!(
                normalize_tensor_name_for(raw, "gemma4").unwrap(),
                canonical,
                "{raw}"
            );
        }
        // Other architectures keep their own naming.
        assert_eq!(
            normalize_tensor_name_for("blk.0.ffn_norm.weight", "qwen2").unwrap(),
            "layers.0.post_attn_norm"
        );
        // Packed tables, the router and fused experts take the names
        // `layered_tensors.rs` derives the requested tensors from.
        let packed = [
            ("per_layer_token_embd.weight", "packed.per_layer_token_embd"),
            ("per_layer_model_proj.weight", "packed.per_layer_model_proj"),
            (
                "blk.5.ffn_gate_up_exps.weight",
                "layers.5.packed.experts_gate_up",
            ),
            ("blk.5.ffn_gate_inp.weight", "layers.5.packed.router_proj"),
            ("blk.5.ffn_gate_inp.scale", "layers.5.packed.router_scale"),
            (
                "blk.5.ffn_down_exps.weight",
                "layers.5.moe.experts_down_proj",
            ),
            ("blk.5.ffn_down_exps.scale", "layers.5.moe.per_expert_scale"),
        ];
        for (raw, canonical) in packed {
            assert_eq!(
                normalize_tensor_name_for(raw, "gemma4").unwrap(),
                canonical,
                "{raw}"
            );
        }
        assert!(matches!(
            normalize_tensor_name_for("blk.5.attn_q.scale", "gemma4"),
            Err(ProductionIngestionError::MalformedMetadata { .. })
        ));
    }
}
