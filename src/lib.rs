//! Production GGUF file ingestion, implementing `magnetar-runtime`'s
//! generic [`ProductionModelArtifactIngestor`] contract
//! (`wire-gguf-into-model-loading`, `production-model-ingestion`
//! capability). Composes the real `magnetar-format-gguf` parser --
//! `magnetar-runtime` never imports this crate (enforced by
//! `submodule-integration`'s dependency guard).
//!
//! Scoped to a single local `model.gguf` file declaring
//! `general.architecture == "qwen2"` (the only architecture family a real
//! Magnetar Model Component exists for). Tensors quantized with GGUF's
//! `Q8_0`/`Q4_K`/`Q5_K` block formats are dequantized to `F32` at
//! payload-read time, *before* the projection-weight transpose runs
//! (`resolve-gguf-quantized-projection-transpose-sequencing`) -- that
//! transpose assumes a flat per-element byte width, meaningless for
//! block-quantized data, so every tensor this crate hands to later steps
//! is genuinely `F32` by the time anything else looks at it, whether or
//! not it started out quantized. GPTQ/AWQ/BitsAndBytes (Hugging Face/
//! Safetensors-shaped quantization schemes, unrelated to GGUF's block
//! format) remain out of scope entirely.
//!
//! Parsing/normalizing a bundle never grants trust (Decision 2, matching
//! `loaders/huggingface`): the returned
//! [`magnetar_runtime::model::ModelManifest`] still goes through
//! `magnetar-runtime`'s own `ModelManifest::validate` and
//! `ModelTrustStore::evaluate` like any other manifest before Model
//! Loading may materialize anything from it.

mod config;
mod dequantize;
mod derived_lm_head;
mod naming;
mod tokenizer;
mod weight_layout;
mod weights;

pub use config::SUPPORTED_ARCHITECTURE;
pub use naming::normalize_tensor_name;
pub use tokenizer::GgufTokenizer;
pub use weights::GgufPayloadSource;

use magnetar_format_gguf::GgufMetadataValue;
use magnetar_runtime::model::{
    ModelArchitecture, ModelArtifactId, ModelArtifactKind, ModelArtifactPart, ModelDigest,
    ModelManifest, ModelName, ModelRevision,
};
use magnetar_runtime::production_model_ingestion::{
    ProductionArtifactPayloadSource, ProductionIngestionError, ProductionIngestionResult,
    ProductionModelArtifactIngestor, ProductionModelSource,
};
use std::{collections::BTreeMap, collections::BTreeSet, fs, sync::Arc};

/// The single GGUF file name this ingestor looks for within a bundle's
/// root -- public so a caller deciding *which* ingestor to use for a given
/// bundle (a format-detection concern this crate itself does not own,
/// matching `production-model-ingestion`'s registry-based dispatch model)
/// can check for its presence without hardcoding the string itself.
pub const GGUF_FILE_NAME: &str = "model.gguf";

/// This ingestor's stable registry identity
/// (`ProductionModelArtifactIngestor::ingestor_id`).
pub const INGESTOR_ID: &str = "gguf-qwen2";

/// The external, pinned production Model Artifact ingestor for GGUF files.
#[derive(Default)]
pub struct GgufIngestor;

impl GgufIngestor {
    pub fn new() -> Self {
        Self
    }
}

fn sanitize_model_name(raw: &str) -> String {
    let sanitized: String = raw
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
                ch
            } else {
                '-'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "gguf-bundle".to_string()
    } else {
        format!("gguf-{sanitized}")
    }
}

/// Re-reads and re-parses `model.gguf`'s own key-value metadata to
/// extract its declared chat template text (`tokenizer.chat_template`),
/// if any -- mirrors `loaders/huggingface`'s own "parts are identity/
/// provenance, not payload" convention: the manifest's `chat_template`
/// field is a part-name reference, never the raw template text, so a
/// caller that wants to actually render it calls this (through the same
/// authorized `source` it already holds) and renders the returned text
/// with whatever `magnetar_runtime::ChatTemplateFormatter` implementation
/// it has available -- `loaders/gguf` does not itself depend on a Jinja2
/// engine (kept out entirely, matching this crate's minimal-duplication
/// design; see this change's design.md).
pub fn load_gguf_chat_template(
    source: &ProductionModelSource,
) -> Result<Option<String>, ProductionIngestionError> {
    let path = source.resolve(GGUF_FILE_NAME).map_err(|_| {
        ProductionIngestionError::RequiredPartMissing {
            part: GGUF_FILE_NAME.to_string(),
        }
    })?;
    let bytes = fs::read(&path).map_err(|error| ProductionIngestionError::RequiredPartMissing {
        part: format!("{} ({error})", path.display()),
    })?;
    let artifact = magnetar_format_gguf::parse(&bytes).map_err(|error| {
        ProductionIngestionError::MalformedMetadata {
            reason: format!("{GGUF_FILE_NAME} failed to parse as GGUF: {error}"),
        }
    })?;
    Ok(match artifact.metadata.get("tokenizer.chat_template") {
        Some(GgufMetadataValue::String(template)) => Some(template.clone()),
        _ => None,
    })
}

/// Re-reads and re-parses `model.gguf`'s own key-value metadata to build a
/// real [`GgufTokenizer`] from its embedded vocabulary -- the GGUF
/// counterpart to a caller separately reading `tokenizer.json` bytes for a
/// Hugging Face bundle: [`ProductionIngestionResult`] never carries a
/// tokenizer (Decision 1's "semantic output" is a manifest plus payload
/// access, not every derived artifact), so any caller building one for a
/// GGUF-sourced model calls this the same way it would separately
/// construct a `HuggingFaceTokenizer` from a Hugging Face bundle's own
/// files.
pub fn load_gguf_tokenizer(
    source: &ProductionModelSource,
    artifact_id: impl Into<String>,
    expected_vocab_size: Option<u64>,
) -> Result<GgufTokenizer, ProductionIngestionError> {
    let path = source.resolve(GGUF_FILE_NAME).map_err(|_| {
        ProductionIngestionError::RequiredPartMissing {
            part: GGUF_FILE_NAME.to_string(),
        }
    })?;
    let bytes = fs::read(&path).map_err(|error| ProductionIngestionError::RequiredPartMissing {
        part: format!("{} ({error})", path.display()),
    })?;
    let artifact = magnetar_format_gguf::parse(&bytes).map_err(|error| {
        ProductionIngestionError::MalformedMetadata {
            reason: format!("{GGUF_FILE_NAME} failed to parse as GGUF: {error}"),
        }
    })?;
    GgufTokenizer::from_gguf_metadata(&artifact.metadata, artifact_id, expected_vocab_size)
}

impl ProductionModelArtifactIngestor for GgufIngestor {
    fn ingestor_id(&self) -> &str {
        INGESTOR_ID
    }

    fn ingest(
        &self,
        source: &ProductionModelSource,
    ) -> Result<ProductionIngestionResult, ProductionIngestionError> {
        let (mut tensors, payload_source, metadata) = weights::discover_and_parse_weights(source)?;
        if tensors.is_empty() {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: "no tensors were discovered in this GGUF file".into(),
            });
        }

        let vocab_size = tensors
            .iter()
            .find(|tensor| tensor.name == "token_embedding")
            .and_then(|tensor| tensor.shape.first().copied())
            .ok_or_else(|| ProductionIngestionError::MalformedMetadata {
                reason: "no token_embedding tensor was discovered".into(),
            })?;
        let mut architecture_config = config::normalize(&metadata, vocab_size)?;

        // Neither real evidence GGUF's key-value metadata reliably
        // declares -- the only real evidence for each is the tensor
        // inventory this ingestor itself just discovered, exactly
        // mirroring `loaders/huggingface`'s identical deferral for the
        // same reason.
        architecture_config.attention_bias = tensors
            .iter()
            .any(|tensor| tensor.name.ends_with("self_attn.q_bias"));
        let tie_word_embeddings = !tensors.iter().any(|tensor| tensor.name == "lm_head");
        architecture_config.tie_word_embeddings = tie_word_embeddings;

        // Every 2D projection weight is stored (and therefore discovered)
        // in Hugging Face's `nn.Linear` `[out_features, in_features]`
        // convention -- GGUF's own `ne[]` order was already reversed back
        // to this in `weights.rs`, and llama.cpp's real GGUF writer
        // applies no further byte-level transpose for Qwen2 (see this
        // change's design.md for the sourced research) -- the Component
        // expects `[in_features, out_features]` instead; see
        // `weight_layout.rs`.
        let transposing_source =
            weight_layout::TransposingPayloadSource::new(payload_source, &tensors);
        weight_layout::swap_declared_projection_shapes(&mut tensors);

        let token_embedding = tensors
            .iter()
            .find(|tensor| tensor.name == "token_embedding")
            .cloned();
        let synthetic_lm_head_added =
            derived_lm_head::append_synthetic_lm_head_if_tied(&mut tensors, tie_word_embeddings);
        let payload_source: Arc<dyn ProductionArtifactPayloadSource> = if synthetic_lm_head_added {
            let token_embedding = token_embedding.expect(
                "append_synthetic_lm_head_if_tied only returns true when a token_embedding \
                 tensor was found",
            );
            Arc::new(
                derived_lm_head::DerivedLmHeadPayloadSource::new(
                    transposing_source,
                    &token_embedding,
                )
                .ok_or_else(|| ProductionIngestionError::MalformedMetadata {
                    reason: "token_embedding is missing the offset/size/shape metadata \
                                 needed to derive a tied lm_head"
                        .into(),
                })?,
            )
        } else {
            Arc::new(transposing_source)
        };

        let storage_dtype = tensors.first().map(|tensor| tensor.storage_dtype);
        let mut supported_compute_dtypes = BTreeSet::new();
        supported_compute_dtypes.insert(magnetar_runtime::model::ModelDType::F32);

        // GGUF is a single unified file (no separate config/weights
        // split like a Hugging Face bundle) -- the same real content
        // digest anchors both the required `ModelConfig`/`ModelWeights`
        // parts `ModelManifest::validate` requires, and the artifact id
        // itself, honestly representing "this manifest is anchored to
        // this exact file's real bytes" without fabricating a fake
        // separate digest for a part boundary GGUF does not have.
        let file_bytes = fs::read(source.resolve(GGUF_FILE_NAME)?).map_err(|error| {
            ProductionIngestionError::RequiredPartMissing {
                part: format!("{GGUF_FILE_NAME} ({error})"),
            }
        })?;
        let file_digest = ModelDigest::sha256(&file_bytes);

        let mut parts = BTreeMap::new();
        parts.insert(
            "gguf_metadata".to_string(),
            ModelArtifactPart {
                name: "gguf_metadata".to_string(),
                kind: ModelArtifactKind::ModelConfig,
                digest: file_digest.clone(),
                size_bytes: Some(file_bytes.len() as u64),
                required: true,
            },
        );
        let weights_identity = {
            let mut entries: Vec<String> = tensors
                .iter()
                .map(|tensor| {
                    format!(
                        "{}:{:?}:{:?}:{:?}",
                        tensor.name, tensor.shape, tensor.offset_bytes, tensor.size_bytes
                    )
                })
                .collect();
            entries.sort();
            ModelDigest::sha256(entries.join("\n").as_bytes())
        };
        parts.insert(
            "weights".to_string(),
            ModelArtifactPart {
                name: "weights".to_string(),
                kind: ModelArtifactKind::ModelWeights,
                digest: weights_identity,
                size_bytes: None,
                required: true,
            },
        );

        let chat_template_reference = match metadata.get("tokenizer.chat_template") {
            Some(GgufMetadataValue::String(template)) => {
                parts.insert(
                    "chat_template".to_string(),
                    ModelArtifactPart {
                        name: "chat_template".to_string(),
                        kind: ModelArtifactKind::ChatTemplate,
                        digest: ModelDigest::sha256(template.as_bytes()),
                        size_bytes: Some(template.len() as u64),
                        required: false,
                    },
                );
                Some("chat_template".to_string())
            }
            _ => None,
        };

        let model_display_name = match metadata.get("general.name") {
            Some(GgufMetadataValue::String(name)) => name.as_str(),
            _ => config::SUPPORTED_ARCHITECTURE,
        };
        let name = sanitize_model_name(model_display_name);
        let id = ModelArtifactId::new(
            ModelArtifactKind::ModelBundle,
            ModelName::new(name).map_err(|error| ProductionIngestionError::MalformedMetadata {
                reason: error.to_string(),
            })?,
            ModelRevision::new("local").map_err(|error| {
                ProductionIngestionError::MalformedMetadata {
                    reason: error.to_string(),
                }
            })?,
            file_digest,
        );

        let manifest = ModelManifest {
            schema_version: magnetar_runtime::model::MODEL_ARTIFACT_SCHEMA_VERSION,
            id,
            architecture: ModelArchitecture::new("qwen", config::SUPPORTED_ARCHITECTURE),
            parts,
            storage_dtype,
            compute_dtype: None,
            supported_compute_dtypes,
            tensors,
            tokenizer: None,
            tokenizer_config: None,
            chat_template: chat_template_reference,
            prompt_template: None,
            generation: None,
            quantization: None,
            shards: Vec::new(),
            runtime_features: BTreeSet::new(),
            memory_features: BTreeSet::new(),
            provider_capabilities: Vec::new(),
            component: None,
            license: None,
            provenance: None,
            signatures: Vec::new(),
            source: Some(source.kind().clone()),
            architecture_config: Some(architecture_config),
            // astorise/Magnetar#75: this ingestor's own real, known output
            // shape -- never inferred, always stamped by the concrete
            // ingestor that produced it.
            artifact_format: magnetar_runtime::model::ArtifactFormat::Gguf,
        };

        Ok(ProductionIngestionResult {
            manifest,
            payload_source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnetar_runtime::ModelArtifactSource;
    use magnetar_runtime::model::ModelTrustStore;
    use magnetar_runtime::tokenizer::Tokenizer as _;

    fn le_u32(value: u32) -> Vec<u8> {
        value.to_le_bytes().to_vec()
    }
    fn le_u64(value: u64) -> Vec<u8> {
        value.to_le_bytes().to_vec()
    }
    fn gguf_string(value: &str) -> Vec<u8> {
        let mut out = le_u64(value.len() as u64);
        out.extend_from_slice(value.as_bytes());
        out
    }
    fn kv_string(key: &str, value: &str) -> Vec<u8> {
        let mut out = gguf_string(key);
        out.extend(le_u32(8));
        out.extend(gguf_string(value));
        out
    }
    fn kv_uint32(key: &str, value: u32) -> Vec<u8> {
        let mut out = gguf_string(key);
        out.extend(le_u32(4));
        out.extend(le_u32(value));
        out
    }
    fn kv_float32(key: &str, value: f32) -> Vec<u8> {
        let mut out = gguf_string(key);
        out.extend(le_u32(6));
        out.extend(value.to_le_bytes());
        out
    }
    fn kv_string_array(key: &str, values: &[&str]) -> Vec<u8> {
        let mut out = gguf_string(key);
        out.extend(le_u32(9)); // GGUF_METADATA_VALUE_TYPE_ARRAY
        out.extend(le_u32(8)); // element type: STRING
        out.extend(le_u64(values.len() as u64));
        for value in values {
            out.extend(gguf_string(value));
        }
        out
    }

    struct TestTensor {
        name: &'static str,
        dimensions: Vec<u64>,
        ggml_type: u32,
        data: Vec<u8>,
    }

    fn build_gguf(kv_entries: &[Vec<u8>], tensors: &[TestTensor], alignment: u64) -> Vec<u8> {
        let mut file = le_u32(0x4655_4747);
        file.extend(le_u32(3));
        file.extend(le_u64(tensors.len() as u64));
        file.extend(le_u64(kv_entries.len() as u64));
        for entry in kv_entries {
            file.extend(entry);
        }
        let mut tensor_info_bytes = Vec::new();
        let mut data_section = Vec::new();
        let mut next_offset = 0_u64;
        for tensor in tensors {
            let padding = (alignment - (next_offset % alignment)) % alignment;
            data_section.extend(std::iter::repeat_n(0u8, padding as usize));
            next_offset += padding;
            let offset = next_offset;
            tensor_info_bytes.extend(gguf_string(tensor.name));
            tensor_info_bytes.extend(le_u32(tensor.dimensions.len() as u32));
            for dimension in &tensor.dimensions {
                tensor_info_bytes.extend(le_u64(*dimension));
            }
            tensor_info_bytes.extend(le_u32(tensor.ggml_type));
            tensor_info_bytes.extend(le_u64(offset));
            data_section.extend(&tensor.data);
            next_offset += tensor.data.len() as u64;
        }
        file.extend(tensor_info_bytes);
        let start_padding = (alignment - (file.len() as u64 % alignment)) % alignment;
        file.extend(std::iter::repeat_n(0u8, start_padding as usize));
        file.extend(data_section);
        file
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// A minimal but complete, real Qwen2 GGUF byte blob: architecture
    /// metadata, an embedded byte-level BPE vocabulary and merges, and
    /// every tensor a single-layer Qwen2 decoder needs (attention, MLP,
    /// norms, and embedding), tied (`output.weight` deliberately absent).
    ///
    /// Dimensions (hidden 2, heads 1, kv_heads 1, intermediate 4, layers
    /// 1, vocab 4) are small enough to hand-write every tensor's real
    /// values, and large enough that a transpose or shape bug on a
    /// non-square projection would not be masked by every dimension
    /// being equal.
    fn tiny_qwen2_gguf() -> Vec<u8> {
        let kvs = vec![
            kv_string("general.architecture", "qwen2"),
            kv_string("general.name", "tiny-qwen2-test"),
            kv_uint32("qwen2.embedding_length", 2),
            kv_uint32("qwen2.feed_forward_length", 4),
            kv_uint32("qwen2.block_count", 1),
            kv_uint32("qwen2.attention.head_count", 1),
            kv_uint32("qwen2.attention.head_count_kv", 1),
            kv_float32("qwen2.attention.layer_norm_rms_epsilon", 1e-6),
            kv_float32("qwen2.rope.freq_base", 10_000.0),
            kv_uint32("tokenizer.ggml.bos_token_id", 0),
            kv_uint32("tokenizer.ggml.eos_token_id", 1),
            kv_string("tokenizer.ggml.model", "gpt2"),
            kv_string_array(
                "tokenizer.ggml.tokens",
                &["<bos>", "<eos>", "hello", "world"],
            ),
            kv_string_array("tokenizer.ggml.merges", &[]),
        ];
        // token_embd: GGUF ne order [hidden=2, vocab=4] -> reversed to
        // [vocab=4, hidden=2] (row-major): rows are each token's 2-value
        // embedding.
        let token_embd: Vec<f32> = (0..8).map(|i| i as f32).collect();
        // attn_q/attn_k/attn_v/attn_output: 2x2 (square -- hidden=2,
        // head_dim*heads=2), stored GGUF-native ne=[in=2, out=2].
        let square_2x2 = vec![1.0f32, 2.0, 3.0, 4.0];
        // ffn_gate/up: GGUF ne=[in=2, out=4] (intermediate=4, hidden=2).
        let ffn_in2_out4: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        // ffn_down: GGUF ne=[in=4, out=2].
        let ffn_in4_out2: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        let norm = vec![1.0f32, 1.0];

        let tensors = [
            TestTensor {
                name: "token_embd.weight",
                dimensions: vec![2, 4],
                ggml_type: 0,
                data: f32_bytes(&token_embd),
            },
            TestTensor {
                name: "blk.0.attn_norm.weight",
                dimensions: vec![2],
                ggml_type: 0,
                data: f32_bytes(&norm),
            },
            TestTensor {
                name: "blk.0.attn_q.weight",
                dimensions: vec![2, 2],
                ggml_type: 0,
                data: f32_bytes(&square_2x2),
            },
            TestTensor {
                name: "blk.0.attn_k.weight",
                dimensions: vec![2, 2],
                ggml_type: 0,
                data: f32_bytes(&square_2x2),
            },
            TestTensor {
                name: "blk.0.attn_v.weight",
                dimensions: vec![2, 2],
                ggml_type: 0,
                data: f32_bytes(&square_2x2),
            },
            TestTensor {
                name: "blk.0.attn_output.weight",
                dimensions: vec![2, 2],
                ggml_type: 0,
                data: f32_bytes(&square_2x2),
            },
            TestTensor {
                name: "blk.0.ffn_norm.weight",
                dimensions: vec![2],
                ggml_type: 0,
                data: f32_bytes(&norm),
            },
            TestTensor {
                name: "blk.0.ffn_gate.weight",
                dimensions: vec![2, 4],
                ggml_type: 0,
                data: f32_bytes(&ffn_in2_out4),
            },
            TestTensor {
                name: "blk.0.ffn_up.weight",
                dimensions: vec![2, 4],
                ggml_type: 0,
                data: f32_bytes(&ffn_in2_out4),
            },
            TestTensor {
                name: "blk.0.ffn_down.weight",
                dimensions: vec![4, 2],
                ggml_type: 0,
                data: f32_bytes(&ffn_in4_out2),
            },
            TestTensor {
                name: "output_norm.weight",
                dimensions: vec![2],
                ggml_type: 0,
                data: f32_bytes(&norm),
            },
        ];
        build_gguf(&kvs, &tensors, 32)
    }

    #[test]
    fn ingests_a_complete_tiny_gguf_file() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(GGUF_FILE_NAME), tiny_qwen2_gguf()).unwrap();
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let result = GgufIngestor::new().ingest(&source).expect("bundle ingests");

        assert!(result.manifest.architecture_config.is_some());
        let config = result.manifest.architecture_config.clone().unwrap();
        assert_eq!(config.hidden_size, 2);
        assert_eq!(config.intermediate_size, 4);
        assert_eq!(config.num_hidden_layers, 1);
        assert_eq!(config.vocab_size, 4);
        assert!(
            config.tie_word_embeddings,
            "no output.weight tensor was declared, so this must be inferred tied"
        );
        assert!(
            !config.attention_bias,
            "no *_bias tensors were declared, so this must default to false"
        );

        let lm_head = result
            .manifest
            .tensors
            .iter()
            .find(|tensor| tensor.name == "lm_head")
            .expect("a synthetic lm_head tensor was derived from token_embedding");
        assert_eq!(
            lm_head.shape,
            vec![2, 4],
            "hidden x vocab, as the Component expects"
        );

        // Parsing/normalizing alone never grants trust (Decision 2).
        let trust = ModelTrustStore::default().evaluate(&result.manifest);
        assert_eq!(
            trust.status(),
            magnetar_runtime::model::ModelTrustStatus::Unknown
        );
        result.manifest.validate().expect("manifest validates");
    }

    /// `load_gguf_tokenizer`/`load_gguf_chat_template`
    /// (`wire-inference-component-to-generic-registry`'s MAG-01 follow-up):
    /// an external caller (`inference-components`) building a tokenizer/
    /// chat formatter for a GGUF-sourced model calls these instead of
    /// re-implementing GGUF metadata parsing itself. Verified against the
    /// same real GGUF bytes `ingests_a_complete_tiny_gguf_file` already
    /// exercises for ingestion, proving both functions genuinely read the
    /// file and delegate to already-tested `GgufTokenizer::from_gguf_
    /// metadata`, not a placeholder.
    #[test]
    fn load_gguf_tokenizer_and_chat_template_read_the_real_file() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(GGUF_FILE_NAME), tiny_qwen2_gguf()).unwrap();
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );

        let tokenizer = load_gguf_tokenizer(&source, "test-tokenizer", Some(4))
            .expect("tokenizer builds from the real embedded GGUF vocabulary");
        assert_eq!(
            tokenizer.metadata().vocabulary_size,
            4,
            "the tokenizer's real vocabulary size must match this fixture's real \
             tokenizer.ggml.tokens array (<bos>, <eos>, hello, world), threaded through \
             load_gguf_tokenizer unchanged"
        );

        // This fixture's own `kvs` (above) declares no
        // `tokenizer.chat_template` key -- a real, meaningful negative
        // case proving the function distinguishes "key absent" from
        // "parse failed", not just that it never errors.
        let chat_template = load_gguf_chat_template(&source)
            .expect("chat template lookup succeeds even when the key is absent");
        assert_eq!(
            chat_template, None,
            "this fixture declares no tokenizer.chat_template key"
        );
    }

    #[test]
    fn transposes_a_non_square_mlp_projection_correctly() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(GGUF_FILE_NAME), tiny_qwen2_gguf()).unwrap();
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let result = GgufIngestor::new().ingest(&source).expect("bundle ingests");

        let gate_proj = result
            .manifest
            .tensors
            .iter()
            .find(|tensor| tensor.name == "layers.0.mlp.gate_proj")
            .unwrap();
        // GGUF ne=[in=2, out=4] reversed to HF-native [out=4, in=2], then
        // swapped back to the Component's expected [in=2, out=4].
        assert_eq!(gate_proj.shape, vec![2, 4]);

        let range = magnetar_runtime::production_model_ingestion::ProductionPayloadRange {
            identity: gate_proj.name.clone(),
            offset: gate_proj.offset_bytes.unwrap(),
            length: gate_proj.size_bytes.unwrap(),
            digest: None,
        };
        let bytes = result.payload_source.read_payload(&range).unwrap();
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        // Reversing GGUF's ne=[2,4] labels the HF-native shape as [4,2]
        // (out=4, in=2) over the SAME raw bytes [1,2,3,4,5,6,7,8]
        // (row-major over 4 rows of 2). The Component-facing transpose
        // (out,in)->(in,out) of that 4x2 matrix [[1,2],[3,4],[5,6],[7,8]]
        // is the 2x4 matrix [[1,3,5,7],[2,4,6,8]].
        assert_eq!(values, vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]);
    }

    /// The exact scenario `resolve-gguf-quantized-projection-transpose-sequencing`
    /// exists to fix: a *quantized*, *non-square* projection weight must
    /// be dequantized before the projection transpose runs, not after --
    /// applying the transpose to raw quantized bytes would silently
    /// corrupt the block structure. Dimensions here (hidden=4,
    /// intermediate=8) are sized to hold exactly one real `Q8_0` block (32
    /// elements) for `ffn_gate` specifically, and are deliberately
    /// non-square so a transpose bug could not hide behind symmetry.
    #[test]
    fn dequantizes_then_transposes_a_non_square_quantized_projection() {
        const HIDDEN: u64 = 4;
        const INTERMEDIATE: u64 = 8;
        let kvs = vec![
            kv_string("general.architecture", "qwen2"),
            kv_uint32("qwen2.embedding_length", HIDDEN as u32),
            kv_uint32("qwen2.feed_forward_length", INTERMEDIATE as u32),
            kv_uint32("qwen2.block_count", 1),
            kv_uint32("qwen2.attention.head_count", 2),
            kv_uint32("qwen2.attention.head_count_kv", 2),
            kv_uint32("tokenizer.ggml.bos_token_id", 0),
            kv_uint32("tokenizer.ggml.eos_token_id", 1),
            kv_string("tokenizer.ggml.model", "gpt2"),
            kv_string_array("tokenizer.ggml.tokens", &["<bos>", "<eos>", "hi", "lo"]),
            kv_string_array("tokenizer.ggml.merges", &[]),
        ];

        let norm = vec![1.0f32; HIDDEN as usize];
        let square = vec![1.0f32; (HIDDEN * HIDDEN) as usize];
        let ffn_up_down = vec![1.0f32; (HIDDEN * INTERMEDIATE) as usize];
        let token_embd = vec![1.0f32; (HIDDEN * 4) as usize];

        // One real Q8_0 block: d = 1.0, qs = 1..=32 (GGUF ne=[in=4,
        // out=8] -- 32 elements, exactly one block).
        let mut ffn_gate_q8 = Vec::with_capacity(34);
        ffn_gate_q8.extend_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
        ffn_gate_q8.extend((1i8..=32).map(|value| value as u8));

        let tensors = [
            TestTensor {
                name: "token_embd.weight",
                dimensions: vec![HIDDEN, 4],
                ggml_type: 0,
                data: f32_bytes(&token_embd),
            },
            TestTensor {
                name: "blk.0.attn_norm.weight",
                dimensions: vec![HIDDEN],
                ggml_type: 0,
                data: f32_bytes(&norm),
            },
            TestTensor {
                name: "blk.0.attn_q.weight",
                dimensions: vec![HIDDEN, HIDDEN],
                ggml_type: 0,
                data: f32_bytes(&square),
            },
            TestTensor {
                name: "blk.0.attn_k.weight",
                dimensions: vec![HIDDEN, HIDDEN],
                ggml_type: 0,
                data: f32_bytes(&square),
            },
            TestTensor {
                name: "blk.0.attn_v.weight",
                dimensions: vec![HIDDEN, HIDDEN],
                ggml_type: 0,
                data: f32_bytes(&square),
            },
            TestTensor {
                name: "blk.0.attn_output.weight",
                dimensions: vec![HIDDEN, HIDDEN],
                ggml_type: 0,
                data: f32_bytes(&square),
            },
            TestTensor {
                name: "blk.0.ffn_norm.weight",
                dimensions: vec![HIDDEN],
                ggml_type: 0,
                data: f32_bytes(&norm),
            },
            // The one quantized, non-square tensor under test.
            TestTensor {
                name: "blk.0.ffn_gate.weight",
                dimensions: vec![HIDDEN, INTERMEDIATE],
                ggml_type: 8, // GGML_TYPE_Q8_0
                data: ffn_gate_q8,
            },
            TestTensor {
                name: "blk.0.ffn_up.weight",
                dimensions: vec![HIDDEN, INTERMEDIATE],
                ggml_type: 0,
                data: f32_bytes(&ffn_up_down),
            },
            TestTensor {
                name: "blk.0.ffn_down.weight",
                dimensions: vec![INTERMEDIATE, HIDDEN],
                ggml_type: 0,
                data: f32_bytes(&ffn_up_down),
            },
            TestTensor {
                name: "output_norm.weight",
                dimensions: vec![HIDDEN],
                ggml_type: 0,
                data: f32_bytes(&norm),
            },
        ];
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join(GGUF_FILE_NAME),
            build_gguf(&kvs, &tensors, 32),
        )
        .unwrap();
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let result = GgufIngestor::new().ingest(&source).expect("bundle ingests");

        let ffn_gate = result
            .manifest
            .tensors
            .iter()
            .find(|tensor| tensor.name == "layers.0.mlp.gate_proj")
            .unwrap();
        assert_eq!(
            ffn_gate.storage_dtype,
            magnetar_runtime::model::ModelDType::F32,
            "a dequantized tensor's declared storage dtype must be overridden to F32"
        );
        assert!(ffn_gate.quantization.is_none());
        // GGUF ne=[in=4, out=8] reversed to HF-native [out=8, in=4], then
        // swapped back to the Component's expected [in=4, out=8].
        assert_eq!(ffn_gate.shape, vec![HIDDEN, INTERMEDIATE]);

        let range = magnetar_runtime::production_model_ingestion::ProductionPayloadRange {
            identity: ffn_gate.name.clone(),
            offset: ffn_gate.offset_bytes.unwrap(),
            length: ffn_gate.size_bytes.unwrap(),
            digest: None,
        };
        let bytes = result.payload_source.read_payload(&range).unwrap();
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        // Dequantized HF-native [out=8, in=4], row-major, is [[1,2,3,4],
        // [5,6,7,8], [9,10,11,12], [13,14,15,16], [17,18,19,20],
        // [21,22,23,24], [25,26,27,28], [29,30,31,32]]. Transposed to
        // [in=4, out=8]: row i = column i of the original (every 4th
        // value starting at i).
        let expected = vec![
            1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 25.0, 29.0, //
            2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, //
            3.0, 7.0, 11.0, 15.0, 19.0, 23.0, 27.0, 31.0, //
            4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0,
        ];
        assert_eq!(values, expected);
    }

    #[test]
    fn loads_a_real_tokenizer_from_embedded_vocabulary() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(GGUF_FILE_NAME), tiny_qwen2_gguf()).unwrap();
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let bytes = fs::read(source.resolve(GGUF_FILE_NAME).unwrap()).unwrap();
        let artifact = magnetar_format_gguf::parse(&bytes).unwrap();
        let tokenizer =
            GgufTokenizer::from_gguf_metadata(&artifact.metadata, "tiny-qwen2-test", Some(4))
                .expect("real tokenizer builds from embedded vocabulary");
        assert_eq!(tokenizer.metadata().vocabulary_size, 4);
        assert!(
            tokenizer
                .metadata()
                .special_token(magnetar_runtime::tokenizer::SpecialTokenKind::Bos)
                .is_some()
        );
    }

    #[test]
    fn rejects_a_non_qwen2_architecture_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let kvs = vec![kv_string("general.architecture", "llama")];
        let tensors = [TestTensor {
            name: "token_embd.weight",
            dimensions: vec![2, 4],
            ggml_type: 0,
            data: f32_bytes(&[0.0; 8]),
        }];
        fs::write(
            dir.path().join(GGUF_FILE_NAME),
            build_gguf(&kvs, &tensors, 32),
        )
        .unwrap();
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let error = match GgufIngestor::new().ingest(&source) {
            Err(error) => error,
            Ok(_) => panic!("expected an unsupported-format error"),
        };
        assert!(matches!(
            error,
            ProductionIngestionError::UnsupportedFormat { .. }
        ));
    }

    #[test]
    fn rejects_a_missing_gguf_file() {
        let dir = tempfile::tempdir().unwrap();
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let error = match GgufIngestor::new().ingest(&source) {
            Err(error) => error,
            Ok(_) => panic!("expected a required-part-missing error"),
        };
        assert!(matches!(
            error,
            ProductionIngestionError::RequiredPartMissing { .. }
        ));
    }

    #[test]
    fn ingestor_id_is_stable() {
        assert_eq!(GgufIngestor::new().ingestor_id(), INGESTOR_ID);
    }
}
