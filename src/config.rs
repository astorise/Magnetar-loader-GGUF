//! Real GGUF key-value metadata normalization into `magnetar-runtime`'s
//! generic [`ModelArchitectureConfig`] (mirrors `loaders/huggingface`'s
//! `config.rs`'s validation discipline exactly, reading GGUF's own
//! `<architecture>.*`/`tokenizer.ggml.*` keys instead of a `config.json`).
//! Scoped to the architectures in [`SUPPORTED_ARCHITECTURES`]; any other
//! declared architecture is rejected structurally rather than guessed at.
//!
//! GGUF key names/types (`general.architecture`,
//! `<arch>.embedding_length`, `<arch>.attention.head_count(_kv)`,
//! `<arch>.attention.layer_norm_rms_epsilon`, `<arch>.rope.freq_base`,
//! `tokenizer.ggml.{bos,eos}_token_id`) match the upstream
//! `ggml-org/llama.cpp` GGUF writer's real conversions, not a guessed
//! schema.

use magnetar_format_gguf::GgufMetadataValue;
use magnetar_runtime::model::{
    ExpertGroupScore, ExpertGroups, ExpertRoutingConfig, ExpertScoring, LatentAttentionConfig,
    LayerSpec, ModelArchitectureConfig, RopeFrequencyScaling,
};
use magnetar_runtime::operator::RopeYarnFrequencyScaling;
use magnetar_runtime::production_model_ingestion::ProductionIngestionError;
use std::collections::BTreeMap;

/// The `general.architecture` values this loader reads. They share one
/// decoder-block key schema (`<arch>.*`); they differ in how llama.cpp's
/// converter stores the attention projections (see
/// [`rope_rows_interleaved`]) and, for `qwen2moe`, in a routed
/// mixture-of-experts MLP ([`mixture_of_experts`]); `gemma4` declares its
/// layers' differing geometry ([`layer_layout`]); `deepseek2` attends through
/// a compressed latent ([`latent_attention`]) and routes experts its own way
/// ([`expert_layers_and_routing`]).
pub const SUPPORTED_ARCHITECTURES: [&str; 5] =
    ["qwen2", "llama", "qwen2moe", "gemma4", "deepseek2"];

/// Whether llama.cpp's converter permuted `architecture`'s query and key
/// projection rows for its interleaved-pair RoPE: it does for `llama`
/// (`LlamaModel.permute`), not for `qwen2`, whose RoPE rotates half-split
/// pairs as the checkpoint stores them. A permuted projection is restored
/// to the checkpoint's own row order at load (`rope_rows.rs`), so every
/// architecture reaches the Model Component in its upstream layout.
pub fn rope_rows_interleaved(architecture: &str) -> bool {
    architecture == "llama"
}

fn malformed(reason: impl Into<String>) -> ProductionIngestionError {
    ProductionIngestionError::MalformedMetadata {
        reason: reason.into(),
    }
}

fn string_value<'a>(
    metadata: &'a BTreeMap<String, GgufMetadataValue>,
    key: &str,
) -> Option<&'a str> {
    match metadata.get(key) {
        Some(GgufMetadataValue::String(value)) => Some(value.as_str()),
        _ => None,
    }
}

/// Accepts `UInt32` or `UInt64` -- llama.cpp's real GGUF writer uses
/// `UINT32` for every count field this loader reads, but this is
/// defensive against a future/alternate writer choosing the wider type
/// for the same logical field.
fn uint_value(metadata: &BTreeMap<String, GgufMetadataValue>, key: &str) -> Option<u64> {
    match metadata.get(key) {
        Some(GgufMetadataValue::UInt32(value)) => Some(u64::from(*value)),
        Some(GgufMetadataValue::UInt64(value)) => Some(*value),
        _ => None,
    }
}

fn float_value(metadata: &BTreeMap<String, GgufMetadataValue>, key: &str) -> Option<f64> {
    match metadata.get(key) {
        Some(GgufMetadataValue::Float32(value)) => Some(f64::from(*value)),
        Some(GgufMetadataValue::Float64(value)) => Some(*value),
        _ => None,
    }
}

/// An unsigned value, or the first element of an array of them (a
/// per-layer key such as `attention.head_count_kv` in a layered file).
fn uint_or_first(metadata: &BTreeMap<String, GgufMetadataValue>, key: &str) -> Option<u64> {
    uint_value(metadata, key).or_else(|| uint_at(metadata, key, 0))
}

/// Element `index` of an unsigned per-layer array, or the scalar the key
/// holds for every layer.
fn uint_at(metadata: &BTreeMap<String, GgufMetadataValue>, key: &str, index: usize) -> Option<u64> {
    match metadata.get(key) {
        Some(GgufMetadataValue::Array(values)) => match values.get(index)? {
            GgufMetadataValue::UInt32(value) => Some(u64::from(*value)),
            GgufMetadataValue::UInt64(value) => Some(*value),
            GgufMetadataValue::Int32(value) => u64::try_from(*value).ok(),
            _ => None,
        },
        _ => uint_value(metadata, key),
    }
}

fn required_uint(
    metadata: &BTreeMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<u64, ProductionIngestionError> {
    uint_value(metadata, key)
        .ok_or_else(|| malformed(format!("GGUF metadata is missing required key '{key}'")))
}

/// The routed mixture of experts a `qwen2moe` file declares, from its
/// `<arch>.expert_count`/`expert_used_count` metadata and the stored
/// shapes of layer 0's stacked experts (`[experts * intermediate,
/// hidden]`) and shared expert (`[shared, hidden]`); `None` for a dense
/// architecture. llama.cpp's `qwen2moe` never renormalizes the chosen
/// experts' probabilities (Hugging Face `norm_topk_prob = false`).
pub fn mixture_of_experts(
    metadata: &BTreeMap<String, GgufMetadataValue>,
    architecture: &str,
    stored_shape: impl Fn(&str) -> Option<Vec<u64>>,
) -> Result<Option<magnetar_runtime::model::MixtureOfExpertsConfig>, ProductionIngestionError> {
    let key = |name: &str| format!("{architecture}.{name}");
    if architecture == "gemma4" {
        // Routed experts alongside the dense MLP, renormalized, no shared
        // expert; their width is declared.
        let Some(experts) = uint_value(metadata, &key("expert_count")).filter(|count| *count > 0)
        else {
            return Ok(None);
        };
        let field = |name: &str| required_uint(metadata, &key(name));
        let narrow = |value: u64, name: &str| {
            u32::try_from(value)
                .map_err(|_| malformed(format!("GGUF metadata's {name} does not fit in a u32")))
        };
        return Ok(Some(magnetar_runtime::model::MixtureOfExpertsConfig {
            experts: narrow(experts, "expert_count")?,
            experts_per_token: narrow(field("expert_used_count")?, "expert_used_count")?,
            expert_intermediate_size: field("expert_feed_forward_length")?,
            shared_expert_intermediate_size: None,
            normalize_top_k: true,
        }));
    }
    if architecture == "deepseek2" {
        // Routed experts after the leading dense layers, each as wide as
        // the declared expert width, and `expert_shared_count` ungated
        // shared experts summed into one of their combined width. A file
        // without routed experts is dense.
        let Some(experts) = uint_value(metadata, &key("expert_count")).filter(|count| *count > 0)
        else {
            return Ok(None);
        };
        let narrow = |value: u64, name: &str| {
            u32::try_from(value)
                .map_err(|_| malformed(format!("GGUF metadata's {name} does not fit in a u32")))
        };
        let width =
            uint_or_first(metadata, &key("expert_feed_forward_length")).ok_or_else(|| {
                malformed(format!(
                    "GGUF metadata is missing required key '{}'",
                    key("expert_feed_forward_length")
                ))
            })?;
        let shared = uint_value(metadata, &key("expert_shared_count")).unwrap_or(0);
        return Ok(Some(magnetar_runtime::model::MixtureOfExpertsConfig {
            experts: narrow(experts, "expert_count")?,
            experts_per_token: narrow(
                required_uint(metadata, &key("expert_used_count"))?,
                "expert_used_count",
            )?,
            expert_intermediate_size: width,
            shared_expert_intermediate_size: (shared > 0).then_some(shared * width),
            normalize_top_k: matches!(
                metadata.get(&key("expert_weights_norm")),
                Some(GgufMetadataValue::Bool(true))
            ),
        }));
    }
    if architecture != "qwen2moe" {
        return Ok(None);
    }
    let count = |name: &str| -> Result<u32, ProductionIngestionError> {
        u32::try_from(required_uint(metadata, &key(name))?)
            .map_err(|_| malformed(format!("GGUF metadata's {name} does not fit in a u32")))
    };
    let experts = count("expert_count")?;
    let experts_per_token = count("expert_used_count")?;
    let rows = |name: &str| {
        stored_shape(name)
            .and_then(|shape| shape.first().copied())
            .ok_or_else(|| malformed(format!("a mixture-of-experts file has no '{name}' tensor")))
    };
    let stacked = rows("layers.0.moe.experts_gate_proj")?;
    if experts == 0 || !stacked.is_multiple_of(u64::from(experts)) {
        return Err(malformed(format!(
            "{stacked} stacked expert rows are not {experts} equal experts"
        )));
    }
    let shared_expert_intermediate_size =
        stored_shape("layers.0.moe.shared_gate_proj").and_then(|shape| shape.first().copied());
    Ok(Some(magnetar_runtime::model::MixtureOfExpertsConfig {
        experts,
        experts_per_token,
        expert_intermediate_size: stacked / u64::from(experts),
        shared_expert_intermediate_size,
        normalize_top_k: false,
    }))
}

/// Multi-head latent attention, when a `deepseek2` file declares
/// `attention.kv_lora_rank` (`express-latent-attention-forms`). llama.cpp's
/// converter writes the expanded head widths as `attention.key_length` and
/// `value_length` in files from before it split `attn_kv_b`, and as
/// `key_length_mla` and `value_length_mla` (beside the latent's own widths
/// in `key_length` and `value_length`) after; the rotated width is
/// `rope.dimension_count`.
pub fn latent_attention(
    metadata: &BTreeMap<String, GgufMetadataValue>,
    architecture: &str,
) -> Result<Option<LatentAttentionConfig>, ProductionIngestionError> {
    let key = |name: &str| format!("{architecture}.{name}");
    if architecture != "deepseek2" {
        return Ok(None);
    }
    let Some(kv_lora_rank) = uint_value(metadata, &key("attention.kv_lora_rank")) else {
        return Ok(None);
    };
    let field = |mla: &str, legacy: &str| {
        uint_value(metadata, &key(mla))
            .or_else(|| uint_value(metadata, &key(legacy)))
            .ok_or_else(|| {
                malformed(format!(
                    "GGUF metadata declares a kv_lora_rank but neither '{}' nor '{}'",
                    key(mla),
                    key(legacy)
                ))
            })
    };
    let key_width = field("attention.key_length_mla", "attention.key_length")?;
    let v_head_dim = field("attention.value_length_mla", "attention.value_length")?;
    let qk_rope_head_dim = required_uint(metadata, &key("rope.dimension_count"))?;
    let qk_nope_head_dim = key_width.checked_sub(qk_rope_head_dim).ok_or_else(|| {
        malformed(format!(
            "GGUF metadata's {key_width}-wide key heads are narrower than their \
             {qk_rope_head_dim} rotated dimensions"
        ))
    })?;
    let rank = |value: u64, name: &str| {
        u32::try_from(value)
            .map_err(|_| malformed(format!("GGUF metadata's {name} does not fit in a u32")))
    };
    Ok(Some(LatentAttentionConfig {
        q_lora_rank: uint_value(metadata, &key("attention.q_lora_rank"))
            .map(|value| rank(value, "q_lora_rank"))
            .transpose()?,
        kv_lora_rank: rank(kv_lora_rank, "kv_lora_rank")?,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        norm_eps: None,
    }))
}

/// A `deepseek2` file's YaRN rotary scaling (`rope.scaling.type` `yarn`),
/// as llama.cpp applies it: the correction range from `yarn_beta_fast`
/// and `yarn_beta_slow` (32 and 1 when absent) rounded outwards, cosines
/// and sines left unscaled, and the softmax scaled by the square of
/// `1 + 0.1 * m * ln(factor)`, where the converter stores `0.1 * m` as
/// `yarn_log_multiplier` (`m` is the checkpoint's `mscale_all_dim`). The
/// Hugging Face form scales the cosines and sines by
/// `mscale(factor, mscale) / mscale(factor, mscale_all_dim)`, which is `1`
/// for the published DeepSeek V2 and V3 configurations (`mscale` equals
/// `mscale_all_dim`); the file does not record `mscale`.
pub fn yarn_scaling(
    metadata: &BTreeMap<String, GgufMetadataValue>,
    architecture: &str,
) -> Result<Option<RopeFrequencyScaling>, ProductionIngestionError> {
    let key = |name: &str| format!("{architecture}.{name}");
    if architecture != "deepseek2"
        || string_value(metadata, &key("rope.scaling.type")) != Some("yarn")
    {
        return Ok(None);
    }
    let required = |name: &str| {
        float_value(metadata, &key(name)).ok_or_else(|| {
            malformed(format!(
                "GGUF metadata's yarn scaling has no '{}'",
                key(name)
            ))
        })
    };
    let scaling = RopeYarnFrequencyScaling {
        factor: required("rope.scaling.factor")?,
        original_context_length: required_uint(
            metadata,
            &key("rope.scaling.original_context_length"),
        )?,
        beta_fast: float_value(metadata, &key("rope.scaling.yarn_beta_fast")).unwrap_or(32.0),
        beta_slow: float_value(metadata, &key("rope.scaling.yarn_beta_slow")).unwrap_or(1.0),
        attention_factor: 1.0,
        truncate: true,
    };
    scaling.validate().map_err(malformed)?;
    Ok(Some(RopeFrequencyScaling::Yarn {
        scaling,
        mscale_all_dim: float_value(metadata, &key("rope.scaling.yarn_log_multiplier"))
            .filter(|multiplier| *multiplier != 0.0)
            .map(|multiplier| multiplier / 0.1),
    }))
}

/// Which layers of a `deepseek2` file route experts, and how
/// (`express-latent-attention-forms`):
/// - the first `leading_dense_block_count` layers are dense;
/// - `expert_gating_func` scores the experts by softmax (`1`, or absent, as
///   llama.cpp assumes for DeepSeek V2 files) or sigmoid (`2`);
/// - `expert_group_count` groups of which `expert_group_used_count` are
///   kept, each scored by its best expert after a softmax and by its two
///   best after a sigmoid, as the checkpoints' own routing does (llama.cpp
///   sums the two best in both cases);
/// - the chosen weights are multiplied by `expert_weights_scale`;
/// - a selection-only bias per expert when the file stores
///   `exp_probs_b` (`has_selection_bias`);
/// - the shared experts are ungated.
pub fn expert_layers_and_routing(
    metadata: &BTreeMap<String, GgufMetadataValue>,
    architecture: &str,
    config: &ModelArchitectureConfig,
    has_selection_bias: bool,
) -> Result<(Option<Vec<bool>>, Option<ExpertRoutingConfig>), ProductionIngestionError> {
    let key = |name: &str| format!("{architecture}.{name}");
    if architecture != "deepseek2" || config.mixture_of_experts.is_none() {
        return Ok((None, None));
    }
    let layers = config.num_hidden_layers as usize;
    let dense = uint_value(metadata, &key("leading_dense_block_count")).unwrap_or(0) as usize;
    if dense >= layers {
        return Err(malformed(format!(
            "GGUF metadata declares routed experts, but all {layers} layers are dense"
        )));
    }
    let moe_layers = (dense > 0).then(|| (0..layers).map(|layer| layer >= dense).collect());
    let scoring = match uint_value(metadata, &key("expert_gating_func")) {
        None | Some(1) => ExpertScoring::Softmax,
        Some(2) => ExpertScoring::Sigmoid,
        Some(other) => {
            return Err(ProductionIngestionError::UnsupportedFormat {
                reason: format!("GGUF metadata's expert_gating_func {other} is not supported"),
            });
        }
    };
    let groups = match uint_value(metadata, &key("expert_group_count")) {
        Some(count) if count > 1 => {
            let narrow = |value: u64, name: &str| {
                u32::try_from(value)
                    .map_err(|_| malformed(format!("GGUF metadata's {name} does not fit in a u32")))
            };
            Some(ExpertGroups {
                count: narrow(count, "expert_group_count")?,
                per_token: narrow(
                    required_uint(metadata, &key("expert_group_used_count"))?,
                    "expert_group_used_count",
                )?,
                score: match scoring {
                    ExpertScoring::Softmax => ExpertGroupScore::Max,
                    ExpertScoring::Sigmoid => ExpertGroupScore::TopTwoSum,
                },
            })
        }
        _ => None,
    };
    Ok((
        moe_layers,
        Some(ExpertRoutingConfig {
            scoring,
            groups,
            routed_scaling_factor: float_value(metadata, &key("expert_weights_scale"))
                .unwrap_or(1.0),
            selection_bias: has_selection_bias,
            normalization: None,
            shared_expert_gated: false,
        }),
    ))
}

/// `general.architecture`, when it is one this loader reads.
pub fn architecture(
    metadata: &BTreeMap<String, GgufMetadataValue>,
) -> Result<&'static str, ProductionIngestionError> {
    let architecture = string_value(metadata, "general.architecture")
        .ok_or_else(|| malformed("GGUF metadata is missing required key 'general.architecture'"))?;
    SUPPORTED_ARCHITECTURES
        .into_iter()
        .find(|supported| *supported == architecture)
        .ok_or_else(|| ProductionIngestionError::UnsupportedFormat {
            reason: format!(
                "GGUF declares architecture '{architecture}'; only {SUPPORTED_ARCHITECTURES:?} \
                 are supported"
            ),
        })
}

/// Normalizes a GGUF file's key-value metadata into a
/// [`ModelArchitectureConfig`], given `vocab_size` (the real embedding
/// table row count, derived from the discovered `token_embedding`
/// tensor's own shape -- GGUF's key-value metadata is not the
/// authoritative source for this value, the tensor inventory is, exactly
/// as a real Hugging Face `config.json`'s declared `vocab_size` is
/// expected to match `model.embed_tokens.weight`'s real shape).
///
/// `attention_bias` is always returned `false` here and is the caller's
/// responsibility to patch based on real tensor discovery (whether
/// `blk.N.attn_{q,k,v}.bias` tensors are actually present) -- mirrors
/// `loaders/huggingface::config::parse`'s identical deferral for the same
/// reason: a real Qwen2/2.5 architectural default is not itself a
/// GGUF-declared field, and real tensor presence is the only genuine
/// evidence this should ever be based on.
pub fn normalize(
    metadata: &BTreeMap<String, GgufMetadataValue>,
    vocab_size: u64,
) -> Result<ModelArchitectureConfig, ProductionIngestionError> {
    let architecture = architecture(metadata)?;
    let key = |name: &str| format!("{architecture}.{name}");

    let hidden_size = required_uint(metadata, &key("embedding_length"))?;
    // A layered file declares some values per layer: the architecture-wide
    // one is the first layer's (`layer_layout` gives every layer's).
    let intermediate_size =
        uint_or_first(metadata, &key("feed_forward_length")).ok_or_else(|| {
            malformed(format!(
                "GGUF metadata is missing required key '{}'",
                key("feed_forward_length")
            ))
        })?;
    let num_hidden_layers = u32::try_from(required_uint(metadata, &key("block_count"))?)
        .map_err(|_| malformed("GGUF metadata's block_count does not fit in a u32"))?;
    let num_attention_heads = u32::try_from(required_uint(metadata, &key("attention.head_count"))?)
        .map_err(|_| malformed("GGUF metadata's attention.head_count does not fit in a u32"))?;
    let num_key_value_heads = match uint_or_first(metadata, &key("attention.head_count_kv")) {
        Some(value) => u32::try_from(value).map_err(|_| {
            malformed("GGUF metadata's attention.head_count_kv does not fit in a u32")
        })?,
        None => num_attention_heads,
    };

    if hidden_size == 0 || intermediate_size == 0 || vocab_size == 0 {
        return Err(malformed(
            "GGUF metadata/tensor inventory declares a zero-valued hidden_size/\
             intermediate_size/vocab_size",
        ));
    }
    // Latent attention expands one K/V head per query head, each
    // `qk_nope + qk_rope` wide (`latent_attention`), whatever the file's
    // MQA-form `head_count_kv` and `key_length` say.
    let latent = latent_attention(metadata, architecture)?;
    let num_key_value_heads = if latent.is_some() {
        num_attention_heads
    } else {
        num_key_value_heads
    };
    if num_hidden_layers == 0 || num_attention_heads == 0 || num_key_value_heads == 0 {
        return Err(malformed(
            "GGUF metadata declares a zero-valued layer/head count",
        ));
    }
    if !num_attention_heads.is_multiple_of(num_key_value_heads) {
        return Err(malformed(format!(
            "GGUF metadata's attention.head_count ({num_attention_heads}) is not an exact \
             multiple of head_count_kv ({num_key_value_heads})"
        )));
    }

    // A sliding-window layer's head width when the file declares one apart
    // (`key_length_swa`): the architecture-wide width of a layered file.
    let head_dim = match latent
        .map(|mla| mla.qk_nope_head_dim + mla.qk_rope_head_dim)
        .or_else(|| uint_value(metadata, &key("attention.key_length_swa")))
        .or_else(|| uint_value(metadata, &key("attention.key_length")))
    {
        Some(declared) => declared,
        None => {
            if !hidden_size.is_multiple_of(u64::from(num_attention_heads)) {
                return Err(malformed(format!(
                    "GGUF metadata declares no attention.key_length and embedding_length \
                     ({hidden_size}) is not evenly divisible by attention.head_count \
                     ({num_attention_heads})"
                )));
            }
            hidden_size / u64::from(num_attention_heads)
        }
    };
    if head_dim == 0 {
        return Err(malformed(
            "GGUF metadata's head_dim (declared or derived) is zero",
        ));
    }

    let rms_norm_eps =
        float_value(metadata, &key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6) as f32;
    let rope_theta = float_value(metadata, &key("rope.freq_base_swa"))
        .or_else(|| float_value(metadata, &key("rope.freq_base")))
        .unwrap_or(10_000.0);
    if rope_theta <= 0.0 {
        return Err(malformed("GGUF metadata's rope.freq_base must be positive"));
    }

    let rope_frequency_scaling = yarn_scaling(metadata, architecture)?;
    let architecture_config = ModelArchitectureConfig {
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        vocab_size,
        rms_norm_eps,
        rope_theta,
        rope_scaling_factor: if rope_frequency_scaling.is_some() {
            None
        } else {
            float_value(metadata, &key("rope.scaling.factor")).map(|value| value as f32)
        },
        // A per-frequency RoPE scaling is stored as a `rope_freqs` tensor,
        // which `lib.rs` rejects; YaRN is declared by keys.
        rope_frequency_scaling,
        // GGUF architecture identity fixes these decoder classes' MLP
        // activation: these GGUF architectures use SiLU, while Gemma 4
        // uses tanh-approximate GELU.
        hidden_activation: match architecture {
            "llama" | "qwen2" | "qwen2moe" | "deepseek2" => Some("silu".to_string()),
            "gemma4" => Some("gelu_pytorch_tanh".to_string()),
            _ => None,
        },
        latent_attention: latent,
        // Patched to real tensor-derived evidence by the caller; see this
        // function's own doc comment.
        tie_word_embeddings: false,
        attention_bias: false,
        mixture_of_experts: None,
        bos_token_id: uint_value(metadata, "tokenizer.ggml.bos_token_id")
            .and_then(|value| u32::try_from(value).ok()),
        eos_token_id: uint_value(metadata, "tokenizer.ggml.eos_token_id")
            .and_then(|value| u32::try_from(value).ok()),
        // `layer_layout` and the `1.6.0` options: uniform layers, none.
        ..ModelArchitectureConfig::default()
    };
    architecture_config.validate().map_err(|error| {
        malformed(format!(
            "GGUF metadata produced an inconsistent architecture config: {error}"
        ))
    })?;

    Ok(architecture_config)
}

/// The checkpoint-wide options of a layered file: its logit softcap and
/// its per-layer input width (`0`: none).
pub fn architecture_options(
    metadata: &BTreeMap<String, GgufMetadataValue>,
    architecture: &str,
) -> (Option<f64>, Option<u32>) {
    let key = |name: &str| format!("{architecture}.{name}");
    (
        float_value(metadata, &key("final_logit_softcapping")),
        uint_value(metadata, &key("embedding_length_per_layer_input"))
            .and_then(|width| u32::try_from(width).ok())
            .filter(|width| *width > 0),
    )
}

/// The per-layer geometry of a file declaring
/// `<arch>.attention.sliding_window_pattern` (`per-layer-kv-geometry`);
/// `None` for uniform layers. Follows llama.cpp's own conversion:
/// - a layer whose pattern entry is `true` attends over `sliding_window`
///   tokens with `key_length_swa`-wide heads and RoPE base
///   `freq_base_swa`; the others over the whole context with
///   `key_length`-wide heads and `freq_base`;
/// - `head_count_kv` and `feed_forward_length` may be per-layer arrays;
/// - a full layer without its own `attn_v` tensor derives V from K
///   (`has_tensor` answers for a canonical name);
/// - `rope_freqs`, when present, holds one factor per rotated pair of a
///   full layer's head: `1` rotates, a huge factor leaves the pair at
///   frequency zero. Its leading ones give the rotated fraction;
/// - the last `shared_kv_layers` layers reuse the K/V of the last earlier
///   layer of their kind.
pub fn layer_layout(
    metadata: &BTreeMap<String, GgufMetadataValue>,
    architecture: &str,
    base: &ModelArchitectureConfig,
    has_tensor: impl Fn(&str) -> bool,
    rope_freqs: Option<&[f32]>,
) -> Result<Option<Vec<LayerSpec>>, ProductionIngestionError> {
    let key = |name: &str| format!("{architecture}.{name}");
    let Some(GgufMetadataValue::Array(pattern)) =
        metadata.get(&key("attention.sliding_window_pattern"))
    else {
        return Ok(None);
    };
    let layers = base.num_hidden_layers as usize;
    if pattern.len() != layers {
        return Err(malformed(format!(
            "GGUF metadata's sliding_window_pattern has {} entries for {layers} layers",
            pattern.len()
        )));
    }
    let sliding: Vec<bool> = pattern
        .iter()
        .map(|entry| match entry {
            GgufMetadataValue::Bool(value) => Ok(*value),
            other => Err(malformed(format!(
                "GGUF metadata's sliding_window_pattern holds {other:?}, not booleans"
            ))),
        })
        .collect::<Result<_, _>>()?;
    let full_head_dim = uint_value(metadata, &key("attention.key_length")).unwrap_or(base.head_dim);
    let sliding_head_dim =
        uint_value(metadata, &key("attention.key_length_swa")).unwrap_or(full_head_dim);
    for (value, keys, width) in [
        (
            "attention.value_length",
            "attention.key_length",
            full_head_dim,
        ),
        (
            "attention.value_length_swa",
            "attention.key_length_swa",
            sliding_head_dim,
        ),
    ] {
        if uint_value(metadata, &key(value)).is_some_and(|value| value != width) {
            return Err(ProductionIngestionError::UnsupportedFormat {
                reason: format!("GGUF metadata's {value} differs from its {keys}"),
            });
        }
    }
    let full_theta = float_value(metadata, &key("rope.freq_base")).unwrap_or(base.rope_theta);
    let sliding_theta = float_value(metadata, &key("rope.freq_base_swa")).unwrap_or(full_theta);
    let window = uint_value(metadata, &key("attention.sliding_window"))
        .map(|window| {
            u32::try_from(window)
                .map_err(|_| malformed("GGUF metadata's sliding_window does not fit in a u32"))
        })
        .transpose()?;
    // The rotated fraction of a full layer's head.
    let partial_rotary_factor = match rope_freqs {
        None => None,
        Some(factors) => {
            let rotated = factors.iter().take_while(|factor| **factor == 1.0).count();
            if factors[rotated..].iter().any(|factor| *factor < 1e20) {
                return Err(ProductionIngestionError::UnsupportedFormat {
                    reason: "GGUF rope_freqs scales frequencies rather than selecting the \
                             rotated pairs, which layered loading does not support"
                        .into(),
                });
            }
            let factor = (2 * rotated) as f64 / full_head_dim as f64;
            (factor != 1.0).then_some(factor)
        }
    };
    let shared = uint_value(metadata, &key("attention.shared_kv_layers")).unwrap_or(0) as usize;
    let first_shared = layers.checked_sub(shared).ok_or_else(|| {
        malformed(format!(
            "GGUF metadata shares the K/V of {shared} of {layers} layers"
        ))
    })?;
    let mut layout = Vec::with_capacity(layers);
    for (index, is_sliding) in sliding.iter().copied().enumerate() {
        let kv_source_layer = if index >= first_shared {
            let producer = sliding[..first_shared]
                .iter()
                .rposition(|producer| *producer == is_sliding)
                .ok_or_else(|| {
                    malformed(format!(
                        "GGUF layer {index} shares the K/V of an earlier layer of its kind, \
                         but none computes its own"
                    ))
                })?;
            Some(producer as u32)
        } else {
            None
        };
        let kv_heads = uint_at(metadata, &key("attention.head_count_kv"), index)
            .unwrap_or(u64::from(base.num_key_value_heads));
        layout.push(LayerSpec {
            sliding_window: if is_sliding {
                Some(window.ok_or_else(|| {
                    malformed("GGUF metadata has sliding layers but no sliding_window")
                })?)
            } else {
                None
            },
            head_dim: if is_sliding {
                sliding_head_dim
            } else {
                full_head_dim
            },
            num_key_value_heads: u32::try_from(kv_heads)
                .map_err(|_| malformed("GGUF metadata's head_count_kv does not fit in a u32"))?,
            rope_theta: if is_sliding {
                sliding_theta
            } else {
                full_theta
            },
            partial_rotary_factor: if is_sliding {
                None
            } else {
                partial_rotary_factor
            },
            value_from_key: !is_sliding
                && kv_source_layer.is_none()
                && !has_tensor(&format!("layers.{index}.self_attn.v_proj")),
            kv_source_layer,
            intermediate_size: uint_at(metadata, &key("feed_forward_length"), index)
                .unwrap_or(base.intermediate_size),
        });
    }
    let uniform = LayerSpec {
        sliding_window: None,
        head_dim: base.head_dim,
        num_key_value_heads: base.num_key_value_heads,
        rope_theta: base.rope_theta,
        partial_rotary_factor: None,
        value_from_key: false,
        kv_source_layer: None,
        intermediate_size: base.intermediate_size,
    };
    Ok((!layout.iter().all(|layer| *layer == uniform)).then_some(layout))
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn qwen2_0_5b_metadata_for_layout() -> BTreeMap<String, GgufMetadataValue> {
        qwen2_0_5b_metadata()
    }

    fn qwen2_0_5b_metadata() -> BTreeMap<String, GgufMetadataValue> {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String("qwen2".into()),
        );
        metadata.insert(
            "qwen2.embedding_length".into(),
            GgufMetadataValue::UInt32(896),
        );
        metadata.insert(
            "qwen2.feed_forward_length".into(),
            GgufMetadataValue::UInt32(4864),
        );
        metadata.insert("qwen2.block_count".into(), GgufMetadataValue::UInt32(24));
        metadata.insert(
            "qwen2.attention.head_count".into(),
            GgufMetadataValue::UInt32(14),
        );
        metadata.insert(
            "qwen2.attention.head_count_kv".into(),
            GgufMetadataValue::UInt32(2),
        );
        metadata.insert(
            "qwen2.attention.layer_norm_rms_epsilon".into(),
            GgufMetadataValue::Float32(1e-6),
        );
        metadata.insert(
            "qwen2.rope.freq_base".into(),
            GgufMetadataValue::Float32(1_000_000.0),
        );
        metadata.insert(
            "tokenizer.ggml.bos_token_id".into(),
            GgufMetadataValue::UInt32(151643),
        );
        metadata.insert(
            "tokenizer.ggml.eos_token_id".into(),
            GgufMetadataValue::UInt32(151645),
        );
        metadata
    }

    #[test]
    fn normalizes_a_real_qwen2_gguf_metadata_set() {
        let config = normalize(&qwen2_0_5b_metadata(), 151936).expect("valid metadata normalizes");
        assert_eq!(config.hidden_size, 896);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.num_attention_heads, 14);
        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.head_dim, 64, "head_dim must be derived: 896 / 14");
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.bos_token_id, Some(151643));
        assert_eq!(config.eos_token_id, Some(151645));
    }

    #[test]
    fn reads_the_same_keys_under_the_llama_architecture() {
        let metadata: BTreeMap<String, GgufMetadataValue> = qwen2_0_5b_metadata()
            .into_iter()
            .map(|(key, value)| match key.strip_prefix("qwen2.") {
                Some(rest) => (format!("llama.{rest}"), value),
                None if key == "general.architecture" => {
                    (key, GgufMetadataValue::String("llama".into()))
                }
                None => (key, value),
            })
            .collect();
        let llama = normalize(&metadata, 151936).unwrap();
        let qwen2 = normalize(&qwen2_0_5b_metadata(), 151936).unwrap();
        assert_eq!(llama, qwen2);
        assert_eq!(llama.hidden_activation.as_deref(), Some("silu"));
        assert!(rope_rows_interleaved(architecture(&metadata).unwrap()));
        assert!(!rope_rows_interleaved(
            architecture(&qwen2_0_5b_metadata()).unwrap()
        ));
    }

    #[test]
    fn rejects_an_unsupported_architecture() {
        let mut metadata = qwen2_0_5b_metadata();
        metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String("gemma3".into()),
        );
        let error = normalize(&metadata, 151936).unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::UnsupportedFormat { .. }
        ));
    }

    #[test]
    fn rejects_missing_required_key() {
        let mut metadata = qwen2_0_5b_metadata();
        metadata.remove("qwen2.embedding_length");
        let error = normalize(&metadata, 151936).unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::MalformedMetadata { .. }
        ));
    }

    #[test]
    fn rejects_invalid_head_kv_relationship() {
        let mut metadata = qwen2_0_5b_metadata();
        metadata.insert(
            "qwen2.attention.head_count_kv".into(),
            GgufMetadataValue::UInt32(3), // 14 % 3 != 0
        );
        let error = normalize(&metadata, 151936).unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::MalformedMetadata { .. }
        ));
    }

    #[test]
    fn rejects_zero_vocab_size() {
        let error = normalize(&qwen2_0_5b_metadata(), 0).unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::MalformedMetadata { .. }
        ));
    }

    #[test]
    fn derives_head_dim_only_when_evenly_divisible() {
        let mut metadata = qwen2_0_5b_metadata();
        metadata.insert(
            "qwen2.embedding_length".into(),
            GgufMetadataValue::UInt32(897), // not divisible by 14
        );
        let error = normalize(&metadata, 151936).unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::MalformedMetadata { .. }
        ));
    }

    #[test]
    fn honors_an_explicit_key_length_over_the_derived_value() {
        let mut metadata = qwen2_0_5b_metadata();
        metadata.insert(
            "qwen2.attention.key_length".into(),
            GgufMetadataValue::UInt32(128),
        );
        let config = normalize(&metadata, 151936).unwrap();
        assert_eq!(config.head_dim, 128);
    }
}

/// `per-layer-kv-geometry`: the metadata of llama.cpp's `gemma4` files, as
/// read from the headers of `google/gemma-4-E2B-it-qat-q4_0-gguf` and
/// `google/gemma-4-26B-A4B-it-qat-q4_0-gguf`. They give the layouts the
/// Hugging Face `config.json` of the same checkpoints gives.
#[cfg(test)]
mod gemma4_tests {
    use super::*;

    fn insert(
        metadata: &mut BTreeMap<String, GgufMetadataValue>,
        key: &str,
        value: GgufMetadataValue,
    ) {
        metadata.insert(format!("gemma4.{key}"), value);
    }

    fn pattern(layers: usize, full_every: usize, first_full: usize) -> GgufMetadataValue {
        GgufMetadataValue::Array(
            (0..layers)
                .map(|index| GgufMetadataValue::Bool(index % full_every != first_full))
                .collect(),
        )
    }

    fn uints(values: impl IntoIterator<Item = u32>) -> GgufMetadataValue {
        GgufMetadataValue::Array(values.into_iter().map(GgufMetadataValue::UInt32).collect())
    }

    fn e2b() -> BTreeMap<String, GgufMetadataValue> {
        use GgufMetadataValue::{Float32, UInt32};
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String("gemma4".into()),
        );
        insert(&mut metadata, "block_count", UInt32(35));
        insert(&mut metadata, "embedding_length", UInt32(1536));
        insert(
            &mut metadata,
            "feed_forward_length",
            uints((0..35).map(|index| if index < 15 { 6144 } else { 12288 })),
        );
        insert(&mut metadata, "attention.head_count", UInt32(8));
        insert(&mut metadata, "attention.head_count_kv", UInt32(1));
        insert(&mut metadata, "rope.freq_base", Float32(1_000_000.0));
        insert(&mut metadata, "rope.freq_base_swa", Float32(10_000.0));
        insert(
            &mut metadata,
            "attention.layer_norm_rms_epsilon",
            Float32(1e-6),
        );
        insert(&mut metadata, "attention.key_length", UInt32(512));
        insert(&mut metadata, "attention.value_length", UInt32(512));
        insert(&mut metadata, "attention.key_length_swa", UInt32(256));
        insert(&mut metadata, "attention.value_length_swa", UInt32(256));
        insert(&mut metadata, "final_logit_softcapping", Float32(30.0));
        insert(&mut metadata, "attention.sliding_window", UInt32(512));
        insert(&mut metadata, "attention.shared_kv_layers", UInt32(20));
        insert(
            &mut metadata,
            "embedding_length_per_layer_input",
            UInt32(256),
        );
        insert(
            &mut metadata,
            "attention.sliding_window_pattern",
            pattern(35, 5, 4),
        );
        metadata
    }

    /// `rope_freqs` of a full layer with 512-wide heads: the first 64 of
    /// 256 pairs rotate.
    fn rope_freqs() -> Vec<f32> {
        (0..256)
            .map(|pair| if pair < 64 { 1.0 } else { 1e30 })
            .collect()
    }

    #[test]
    fn an_e2b_file_slides_shares_kv_and_rotates_a_quarter_of_full_heads() {
        let metadata = e2b();
        let base = normalize(&metadata, 262_144).unwrap();
        assert_eq!(base.hidden_activation.as_deref(), Some("gelu_pytorch_tanh"));
        assert_eq!(base.head_dim, 256);
        assert_eq!(base.num_key_value_heads, 1);
        assert_eq!(base.intermediate_size, 6144);
        assert_eq!(base.rope_theta, 10_000.0);
        let layout = layer_layout(&metadata, "gemma4", &base, |_| true, Some(&rope_freqs()))
            .unwrap()
            .unwrap();
        assert_eq!(layout.len(), 35);
        assert_eq!(layout[0].sliding_window, Some(512));
        assert_eq!(layout[0].head_dim, 256);
        assert_eq!(layout[0].rope_theta, 10_000.0);
        assert_eq!(layout[0].partial_rotary_factor, None);
        assert_eq!(layout[4].sliding_window, None);
        assert_eq!(layout[4].head_dim, 512);
        assert_eq!(layout[4].rope_theta, 1_000_000.0);
        assert_eq!(layout[4].partial_rotary_factor, Some(0.25));
        assert!(!layout[4].value_from_key);
        assert_eq!(layout[15].kv_source_layer, Some(13));
        assert_eq!(layout[19].kv_source_layer, Some(14));
        assert_eq!(layout[15].intermediate_size, 12288);
        assert_eq!(
            architecture_options(&metadata, "gemma4"),
            (Some(30.0), Some(256))
        );
        let mut config = base;
        config.layer_layout = Some(layout);
        config.validate().unwrap();
    }

    #[test]
    fn a_full_layer_without_v_projection_derives_v_from_k() {
        use GgufMetadataValue::UInt32;
        let mut metadata = e2b();
        insert(&mut metadata, "block_count", UInt32(30));
        insert(&mut metadata, "embedding_length", UInt32(2816));
        insert(&mut metadata, "feed_forward_length", UInt32(2112));
        insert(&mut metadata, "attention.head_count", UInt32(16));
        insert(
            &mut metadata,
            "attention.head_count_kv",
            uints((0..30).map(|index| if index % 6 == 5 { 2 } else { 8 })),
        );
        insert(&mut metadata, "attention.sliding_window", UInt32(1024));
        insert(&mut metadata, "attention.shared_kv_layers", UInt32(0));
        insert(&mut metadata, "embedding_length_per_layer_input", UInt32(0));
        insert(
            &mut metadata,
            "attention.sliding_window_pattern",
            pattern(30, 6, 5),
        );
        insert(&mut metadata, "expert_count", UInt32(128));
        insert(&mut metadata, "expert_used_count", UInt32(8));
        insert(&mut metadata, "expert_feed_forward_length", UInt32(704));
        let base = normalize(&metadata, 262_144).unwrap();
        assert_eq!(base.num_key_value_heads, 8);
        // The full layers' files have no `attn_v`.
        let has_tensor = |name: &str| {
            !["5", "11", "17", "23", "29"]
                .iter()
                .any(|index| name == format!("layers.{index}.self_attn.v_proj"))
        };
        let layout = layer_layout(&metadata, "gemma4", &base, has_tensor, Some(&rope_freqs()))
            .unwrap()
            .unwrap();
        for (index, layer) in layout.iter().enumerate() {
            let full = index % 6 == 5;
            assert_eq!(layer.value_from_key, full, "layer {index}");
            assert_eq!(layer.num_key_value_heads, if full { 2 } else { 8 });
            assert_eq!(layer.kv_source_layer, None);
        }
        assert_eq!(
            architecture_options(&metadata, "gemma4"),
            (Some(30.0), None)
        );
        assert_eq!(
            mixture_of_experts(&metadata, "gemma4", |_| None).unwrap(),
            Some(magnetar_runtime::model::MixtureOfExpertsConfig {
                experts: 128,
                experts_per_token: 8,
                expert_intermediate_size: 704,
                shared_expert_intermediate_size: None,
                normalize_top_k: true,
            })
        );
    }

    #[test]
    fn rope_freqs_scaling_frequencies_is_refused() {
        let metadata = e2b();
        let base = normalize(&metadata, 262_144).unwrap();
        let scaling: Vec<f32> = (0..256).map(|pair| 1.0 + pair as f32).collect();
        assert!(matches!(
            layer_layout(&metadata, "gemma4", &base, |_| true, Some(&scaling)),
            Err(ProductionIngestionError::UnsupportedFormat { .. })
        ));
    }

    #[test]
    fn a_file_without_sliding_pattern_keeps_uniform_layers() {
        let metadata = super::tests::qwen2_0_5b_metadata_for_layout();
        let base = normalize(&metadata, 151_936).unwrap();
        assert_eq!(
            layer_layout(&metadata, "qwen2", &base, |_| true, None).unwrap(),
            None
        );
    }
}

/// `express-latent-attention-forms`: the metadata of llama.cpp's
/// `deepseek2` files.
#[cfg(test)]
mod deepseek2_tests {
    use super::*;
    use GgufMetadataValue::{Float32, String as Text, UInt32};

    /// The metadata of `DeepSeek-V2-Lite-Chat.Q4_K_M.gguf`
    /// (mradermacher, converted in 2024, before llama.cpp split
    /// `attn_kv_b`), as read from the file's header.
    fn v2_lite() -> BTreeMap<String, GgufMetadataValue> {
        let mut metadata =
            BTreeMap::from([("general.architecture".to_string(), Text("deepseek2".into()))]);
        for (key, value) in [
            ("block_count", UInt32(27)),
            ("context_length", UInt32(163_840)),
            ("embedding_length", UInt32(2048)),
            ("feed_forward_length", UInt32(10_944)),
            ("attention.head_count", UInt32(16)),
            ("attention.head_count_kv", UInt32(16)),
            ("rope.freq_base", Float32(10_000.0)),
            ("attention.layer_norm_rms_epsilon", Float32(1e-6)),
            ("expert_used_count", UInt32(6)),
            ("leading_dense_block_count", UInt32(1)),
            ("vocab_size", UInt32(102_400)),
            ("attention.kv_lora_rank", UInt32(512)),
            ("attention.key_length", UInt32(192)),
            ("attention.value_length", UInt32(128)),
            ("expert_feed_forward_length", UInt32(1408)),
            ("expert_count", UInt32(64)),
            ("expert_shared_count", UInt32(2)),
            ("expert_weights_scale", Float32(1.0)),
            ("rope.dimension_count", UInt32(64)),
            ("rope.scaling.type", Text("yarn".into())),
            ("rope.scaling.factor", Float32(40.0)),
            ("rope.scaling.original_context_length", UInt32(4096)),
            ("rope.scaling.yarn_log_multiplier", Float32(0.0707)),
        ] {
            metadata.insert(format!("deepseek2.{key}"), value);
        }
        metadata
    }

    fn routing(
        metadata: &BTreeMap<String, GgufMetadataValue>,
        selection_bias: bool,
    ) -> Result<(Option<Vec<bool>>, Option<ExpertRoutingConfig>), ProductionIngestionError> {
        let mut config = normalize(metadata, 102_400)?;
        config.mixture_of_experts = mixture_of_experts(metadata, "deepseek2", |_| None)?;
        expert_layers_and_routing(metadata, "deepseek2", &config, selection_bias)
    }

    #[test]
    fn a_v2_lite_file_declares_latent_attention_yarn_and_its_expert_layers() {
        let metadata = v2_lite();
        let config = normalize(&metadata, 102_400).unwrap();
        assert_eq!(
            config.latent_attention,
            Some(LatentAttentionConfig {
                q_lora_rank: None,
                kv_lora_rank: 512,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
                norm_eps: None,
            })
        );
        assert_eq!((config.head_dim, config.num_key_value_heads), (192, 16));
        assert_eq!(config.rope_scaling_factor, None);
        let Some(RopeFrequencyScaling::Yarn {
            scaling,
            mscale_all_dim,
        }) = config.rope_frequency_scaling
        else {
            panic!("no YaRN: {:?}", config.rope_frequency_scaling);
        };
        assert_eq!(
            (scaling.factor, scaling.original_context_length),
            (40.0, 4096)
        );
        assert_eq!((scaling.beta_fast, scaling.beta_slow), (32.0, 1.0));
        assert_eq!((scaling.attention_factor, scaling.truncate), (1.0, true));
        // `0.1 * 0.707` stored in `f32`.
        assert!((mscale_all_dim.unwrap() - 0.707).abs() < 1e-6);
        assert_eq!(
            mixture_of_experts(&metadata, "deepseek2", |_| None).unwrap(),
            Some(magnetar_runtime::model::MixtureOfExpertsConfig {
                experts: 64,
                experts_per_token: 6,
                expert_intermediate_size: 1408,
                shared_expert_intermediate_size: Some(2816),
                normalize_top_k: false,
            })
        );
        let (layers, routing) = routing(&metadata, false).unwrap();
        let layers = layers.unwrap();
        assert_eq!(layers.len(), 27);
        assert!(!layers[0] && layers[1..].iter().all(|routed| *routed));
        assert_eq!(
            routing,
            Some(ExpertRoutingConfig {
                scoring: ExpertScoring::Softmax,
                groups: None,
                routed_scaling_factor: 1.0,
                selection_bias: false,
                normalization: None,
                shared_expert_gated: false,
            })
        );
    }

    /// A file converted after llama.cpp split `attn_kv_b` declares the MQA
    /// form's widths beside the expanded ones.
    #[test]
    fn a_split_key_value_file_reads_the_expanded_head_widths() {
        let mut metadata = v2_lite();
        for (key, value) in [
            ("attention.head_count_kv", 1),
            ("attention.key_length", 576),
            ("attention.value_length", 512),
            ("attention.key_length_mla", 192),
            ("attention.value_length_mla", 128),
        ] {
            metadata.insert(format!("deepseek2.{key}"), UInt32(value));
        }
        let split = normalize(&metadata, 102_400).unwrap();
        let fused = normalize(&v2_lite(), 102_400).unwrap();
        assert_eq!(split.latent_attention, fused.latent_attention);
        assert_eq!((split.head_dim, split.num_key_value_heads), (192, 16));
    }

    /// A sigmoid router with groups scores each group by its two best
    /// experts; a softmax router by its best.
    #[test]
    fn grouped_routers_score_groups_like_their_checkpoints() {
        let mut metadata = v2_lite();
        for (key, value) in [
            ("expert_count", UInt32(256)),
            ("expert_used_count", UInt32(8)),
            ("expert_group_count", UInt32(8)),
            ("expert_group_used_count", UInt32(4)),
            ("expert_gating_func", UInt32(2)),
            ("expert_weights_scale", Float32(2.5)),
        ] {
            metadata.insert(format!("deepseek2.{key}"), value);
        }
        metadata.insert(
            "deepseek2.expert_weights_norm".into(),
            GgufMetadataValue::Bool(true),
        );
        let routing_of = |metadata: &BTreeMap<_, _>| routing(metadata, true).unwrap().1.unwrap();
        let sigmoid = routing_of(&metadata);
        assert_eq!(sigmoid.scoring, ExpertScoring::Sigmoid);
        assert_eq!(
            sigmoid.groups,
            Some(ExpertGroups {
                count: 8,
                per_token: 4,
                score: ExpertGroupScore::TopTwoSum,
            })
        );
        assert_eq!(sigmoid.routed_scaling_factor, 2.5);
        assert!(sigmoid.selection_bias);
        assert!(
            mixture_of_experts(&metadata, "deepseek2", |_| None)
                .unwrap()
                .unwrap()
                .normalize_top_k
        );
        metadata.insert("deepseek2.expert_gating_func".into(), UInt32(1));
        assert_eq!(
            routing_of(&metadata).groups.unwrap().score,
            ExpertGroupScore::Max
        );
        metadata.insert("deepseek2.expert_gating_func".into(), UInt32(3));
        assert!(matches!(
            routing(&metadata, true),
            Err(ProductionIngestionError::UnsupportedFormat { .. })
        ));
    }

    #[test]
    fn incomplete_latent_attention_or_yarn_metadata_is_refused() {
        let mut metadata = v2_lite();
        metadata.remove("deepseek2.rope.dimension_count");
        assert!(normalize(&metadata, 102_400).is_err());
        let mut metadata = v2_lite();
        metadata.remove("deepseek2.rope.scaling.original_context_length");
        assert!(normalize(&metadata, 102_400).is_err());
        let mut metadata = v2_lite();
        metadata.insert("deepseek2.attention.key_length".into(), UInt32(32));
        assert!(normalize(&metadata, 102_400).is_err());
        // Every layer dense, yet routed experts declared.
        let mut metadata = v2_lite();
        metadata.insert("deepseek2.leading_dense_block_count".into(), UInt32(27));
        assert!(routing(&metadata, false).is_err());
    }
}
