//! Real GGUF key-value metadata normalization into `magnetar-runtime`'s
//! generic [`ModelArchitectureConfig`] (mirrors `loaders/huggingface`'s
//! `config.rs`'s validation discipline exactly, reading GGUF's own
//! `<architecture>.*`/`tokenizer.ggml.*` keys instead of a `config.json`).
//! Scoped to `general.architecture == "qwen2"` -- the only architecture
//! family a real Magnetar Model Component (`components/qwen`) exists for
//! today; any other declared architecture is rejected structurally rather
//! than guessed at.
//!
//! GGUF key names/types (`general.architecture`, `qwen2.embedding_length`,
//! `qwen2.attention.head_count(_kv)`, `qwen2.attention.layer_norm_rms_epsilon`,
//! `qwen2.rope.freq_base`, `tokenizer.ggml.{bos,eos}_token_id`) match the
//! upstream `ggml-org/llama.cpp` GGUF writer's real Qwen2 conversion
//! (`conversion/qwen.py`'s `Qwen2Model`), not a guessed schema.

use magnetar_format_gguf::GgufMetadataValue;
use magnetar_runtime::model::ModelArchitectureConfig;
use magnetar_runtime::production_model_ingestion::ProductionIngestionError;
use std::collections::BTreeMap;

pub const SUPPORTED_ARCHITECTURE: &str = "qwen2";

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

fn required_uint(
    metadata: &BTreeMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<u64, ProductionIngestionError> {
    uint_value(metadata, key)
        .ok_or_else(|| malformed(format!("GGUF metadata is missing required key '{key}'")))
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
    let architecture = string_value(metadata, "general.architecture")
        .ok_or_else(|| malformed("GGUF metadata is missing required key 'general.architecture'"))?;
    if architecture != SUPPORTED_ARCHITECTURE {
        return Err(ProductionIngestionError::UnsupportedFormat {
            reason: format!(
                "GGUF declares architecture '{architecture}'; only '{SUPPORTED_ARCHITECTURE}' \
                 has a real production Model Component today"
            ),
        });
    }

    let hidden_size = required_uint(metadata, "qwen2.embedding_length")?;
    let intermediate_size = required_uint(metadata, "qwen2.feed_forward_length")?;
    let num_hidden_layers = u32::try_from(required_uint(metadata, "qwen2.block_count")?)
        .map_err(|_| malformed("GGUF metadata's 'qwen2.block_count' does not fit in a u32"))?;
    let num_attention_heads = u32::try_from(required_uint(metadata, "qwen2.attention.head_count")?)
        .map_err(|_| {
            malformed("GGUF metadata's 'qwen2.attention.head_count' does not fit in a u32")
        })?;
    let num_key_value_heads = match uint_value(metadata, "qwen2.attention.head_count_kv") {
        Some(value) => u32::try_from(value).map_err(|_| {
            malformed("GGUF metadata's 'qwen2.attention.head_count_kv' does not fit in a u32")
        })?,
        None => num_attention_heads,
    };

    if hidden_size == 0 || intermediate_size == 0 || vocab_size == 0 {
        return Err(malformed(
            "GGUF metadata/tensor inventory declares a zero-valued hidden_size/\
             intermediate_size/vocab_size",
        ));
    }
    if num_hidden_layers == 0 || num_attention_heads == 0 || num_key_value_heads == 0 {
        return Err(malformed(
            "GGUF metadata declares a zero-valued layer/head count",
        ));
    }
    if !num_attention_heads.is_multiple_of(num_key_value_heads) {
        return Err(malformed(format!(
            "GGUF metadata's qwen2.attention.head_count ({num_attention_heads}) is not an exact \
             multiple of head_count_kv ({num_key_value_heads})"
        )));
    }

    let head_dim = match uint_value(metadata, "qwen2.attention.key_length") {
        Some(declared) => declared,
        None => {
            if !hidden_size.is_multiple_of(u64::from(num_attention_heads)) {
                return Err(malformed(format!(
                    "GGUF metadata declares no qwen2.attention.key_length and embedding_length \
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
        float_value(metadata, "qwen2.attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f32;
    let rope_theta = float_value(metadata, "qwen2.rope.freq_base").unwrap_or(10_000.0);
    if rope_theta <= 0.0 {
        return Err(malformed(
            "GGUF metadata's qwen2.rope.freq_base must be positive",
        ));
    }

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
        rope_scaling_factor: float_value(metadata, "qwen2.rope.scaling.factor")
            .map(|value| value as f32),
        // Patched to real tensor-derived evidence by the caller; see this
        // function's own doc comment.
        tie_word_embeddings: false,
        attention_bias: false,
        bos_token_id: uint_value(metadata, "tokenizer.ggml.bos_token_id")
            .and_then(|value| u32::try_from(value).ok()),
        eos_token_id: uint_value(metadata, "tokenizer.ggml.eos_token_id")
            .and_then(|value| u32::try_from(value).ok()),
    };
    architecture_config.validate().map_err(|error| {
        malformed(format!(
            "GGUF metadata produced an inconsistent architecture config: {error}"
        ))
    })?;

    Ok(architecture_config)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn rejects_a_non_qwen2_architecture() {
        let mut metadata = qwen2_0_5b_metadata();
        metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String("llama".into()),
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
