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
//! `Q8_0`/`Q4_K`/`Q5_K` block formats keep their original bytes and
//! storage dtype; quantized projections declare transposed storage instead
//! of being transposed byte by byte (`execute-direct-quantized-weights`),
//! and Model Loading converts or hands them over packed. GPTQ/AWQ/BitsAndBytes (Hugging Face/
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
mod derived_lm_head;
mod layered_tensors;
mod naming;
mod quant_blocks;
mod rope_rows;
mod tokenizer;
mod weight_layout;
mod weights;

pub use config::SUPPORTED_ARCHITECTURES;
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
    let artifact = weights::parse_header(&path)?;
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
    let artifact = weights::parse_header(&path)?;
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
        let architecture = config::architecture(&metadata)?;
        let mut architecture_config = config::normalize(&metadata, vocab_size)?;

        // Neither real evidence GGUF's key-value metadata reliably
        // declares -- the only real evidence for each is the tensor
        // inventory this ingestor itself just discovered, exactly
        // mirroring `loaders/huggingface`'s identical deferral for the
        // same reason.
        architecture_config.attention_bias = tensors
            .iter()
            .any(|tensor| tensor.name.ends_with("self_attn.q_bias"));
        // A layered file (`per-layer-kv-geometry`): its layers' geometry,
        // with the rotated fraction read from its `rope_freqs` table, which
        // is not a weight.
        let rope_freqs = match tensors
            .iter()
            .position(|tensor| tensor.name == "rope_freqs")
        {
            Some(index) => {
                let tensor = tensors.remove(index);
                Some(read_f32_tensor(&payload_source, &tensor)?)
            }
            None => None,
        };
        architecture_config.layer_layout = config::layer_layout(
            &metadata,
            architecture,
            &architecture_config,
            |name| tensors.iter().any(|tensor| tensor.name == name),
            rope_freqs.as_deref(),
        )?;
        (
            architecture_config.final_logit_softcapping,
            architecture_config.per_layer_input_size,
        ) = config::architecture_options(&metadata, architecture);
        architecture_config.mixture_of_experts =
            config::mixture_of_experts(&metadata, architecture, |name| {
                tensors
                    .iter()
                    .find(|tensor| tensor.name == name)
                    .map(|tensor| tensor.shape.clone())
            })?;
        (
            architecture_config.moe_layers,
            architecture_config.expert_routing,
        ) = config::expert_layers_and_routing(
            &metadata,
            architecture,
            &architecture_config,
            tensors
                .iter()
                .any(|tensor| tensor.name.ends_with(".moe.selection_bias")),
        )?;
        architecture_config.validate().map_err(|error| {
            ProductionIngestionError::MalformedMetadata {
                reason: format!("GGUF layer layout or mixture of experts is inconsistent: {error}"),
            }
        })?;
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
        // A layered file's packed per-layer tables and routed experts, cut
        // into the tensors its Component requests (`layered_tensors.rs`),
        // still in `[out, in]` orientation.
        let payload_source =
            layered_tensors::derive(&mut tensors, payload_source, &architecture_config)?;
        let payload_source = rope_rows::RopeRowRestoringPayloadSource::new(
            payload_source,
            &mut tensors,
            config::rope_rows_interleaved(architecture),
            architecture_config.head_dim,
        )?;
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
        let file_path = source.resolve(GGUF_FILE_NAME)?;
        let file_length = fs::metadata(&file_path)
            .map_err(|error| ProductionIngestionError::RequiredPartMissing {
                part: format!("{GGUF_FILE_NAME} ({error})"),
            })?
            .len();
        let file_digest = weights::file_digest(&file_path)?;

        let mut parts = BTreeMap::new();
        parts.insert(
            "gguf_metadata".to_string(),
            ModelArtifactPart {
                name: "gguf_metadata".to_string(),
                kind: ModelArtifactKind::ModelConfig,
                digest: file_digest.clone(),
                size_bytes: Some(file_length),
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
            _ => architecture,
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
            // The family is the checkpoint's own architecture name
            // (`general.architecture`), as `loaders/huggingface` uses its
            // `model_type`: one architecture names one family whichever
            // format it came in, so the Model Component that declares it
            // serves both.
            architecture: ModelArchitecture::new(architecture, architecture),
            parts,
            storage_dtype,
            compute_dtype: None,
            supported_compute_dtypes,
            tensors,
            tokenizer: None,
            tokenizer_config: None,
            chat_template: chat_template_reference,
            prompt_template: None,
            artifact_format: magnetar_runtime::model::ArtifactFormat::Gguf,
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
        };

        Ok(ProductionIngestionResult {
            manifest,
            payload_source,
        })
    }
}

/// The values of a small `F32` tensor, read through `source`.
fn read_f32_tensor(
    source: &impl ProductionArtifactPayloadSource,
    tensor: &magnetar_runtime::model::ModelTensorMetadata,
) -> Result<Vec<f32>, ProductionIngestionError> {
    let malformed = |reason: String| ProductionIngestionError::MalformedMetadata { reason };
    if tensor.storage_dtype != magnetar_runtime::model::ModelDType::F32 {
        return Err(malformed(format!(
            "tensor '{}' is stored as {:?}, not F32",
            tensor.name, tensor.storage_dtype
        )));
    }
    let (Some(offset), Some(length)) = (tensor.offset_bytes, tensor.size_bytes) else {
        return Err(malformed(format!(
            "tensor '{}' has no byte range",
            tensor.name
        )));
    };
    let bytes = source.read_payload(
        &magnetar_runtime::production_model_ingestion::ProductionPayloadRange {
            identity: tensor.name.clone(),
            offset,
            length,
            digest: None,
        },
    )?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
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

    fn kv_array(key: &str, element_type: u32, values: &[Vec<u8>]) -> Vec<u8> {
        let mut out = gguf_string(key);
        out.extend(le_u32(9));
        out.extend(le_u32(element_type));
        out.extend(le_u64(values.len() as u64));
        for value in values {
            out.extend(value);
        }
        out
    }

    /// A two-layer `gemma4` file: layer 0 slides over 8 tokens with
    /// 2-wide heads and its own V; layer 1 attends over the whole context
    /// with 4-wide heads, half of each rotated (`rope_freqs` `[1, 1e30]`),
    /// and no `attn_v`.
    fn tiny_gemma4_gguf() -> Vec<u8> {
        gemma4_gguf(0, Vec::new(), Vec::new())
    }

    /// [`tiny_gemma4_gguf`] with per-layer inputs `per_layer_width` wide,
    /// and `extra_kvs` and `extra_tensors` added.
    fn gemma4_gguf(
        per_layer_width: u32,
        extra_kvs: Vec<Vec<u8>>,
        extra_tensors: Vec<TestTensor>,
    ) -> Vec<u8> {
        let mut kvs = vec![
            kv_string("general.architecture", "gemma4"),
            kv_string("general.name", "tiny-gemma4-test"),
            kv_uint32("gemma4.embedding_length", 2),
            kv_array("gemma4.feed_forward_length", 4, &[le_u32(4), le_u32(4)]),
            kv_uint32("gemma4.block_count", 2),
            kv_uint32("gemma4.attention.head_count", 1),
            kv_array("gemma4.attention.head_count_kv", 4, &[le_u32(1), le_u32(1)]),
            kv_float32("gemma4.attention.layer_norm_rms_epsilon", 1e-6),
            kv_float32("gemma4.rope.freq_base", 1_000_000.0),
            kv_float32("gemma4.rope.freq_base_swa", 10_000.0),
            kv_uint32("gemma4.attention.key_length", 4),
            kv_uint32("gemma4.attention.value_length", 4),
            kv_uint32("gemma4.attention.key_length_swa", 2),
            kv_uint32("gemma4.attention.value_length_swa", 2),
            kv_uint32("gemma4.attention.sliding_window", 8),
            kv_uint32("gemma4.attention.shared_kv_layers", 0),
            kv_uint32("gemma4.embedding_length_per_layer_input", per_layer_width),
            kv_float32("gemma4.final_logit_softcapping", 30.0),
            kv_array(
                "gemma4.attention.sliding_window_pattern",
                7,
                &[vec![1], vec![0]],
            ),
            kv_uint32("tokenizer.ggml.bos_token_id", 0),
            kv_uint32("tokenizer.ggml.eos_token_id", 1),
            kv_string("tokenizer.ggml.model", "gpt2"),
            kv_string_array(
                "tokenizer.ggml.tokens",
                &["<bos>", "<eos>", "hello", "world"],
            ),
            kv_string_array("tokenizer.ggml.merges", &[]),
        ];
        let tensor = |name: &'static str, dimensions: Vec<u64>| {
            let count: u64 = dimensions.iter().product();
            TestTensor {
                name,
                dimensions,
                ggml_type: 0,
                data: f32_bytes(&(0..count).map(|i| i as f32 * 0.1 + 0.5).collect::<Vec<_>>()),
            }
        };
        let mut tensors = vec![
            tensor("token_embd.weight", vec![2, 4]),
            tensor("output_norm.weight", vec![2]),
            TestTensor {
                name: "rope_freqs.weight",
                dimensions: vec![2],
                ggml_type: 0,
                data: f32_bytes(&[1.0, 1e30]),
            },
        ];
        // GGUF `ne` order: `[in, out]`.
        for (layer, head) in [(0, 2u64), (1, 4u64)] {
            let name = |field: &str| -> &'static str {
                Box::leak(format!("blk.{layer}.{field}.weight").into_boxed_str())
            };
            tensors.extend([
                tensor(name("attn_norm"), vec![2]),
                tensor(name("attn_q"), vec![2, head]),
                tensor(name("attn_k"), vec![2, head]),
                tensor(name("attn_q_norm"), vec![head]),
                tensor(name("attn_k_norm"), vec![head]),
                tensor(name("attn_output"), vec![head, 2]),
                tensor(name("post_attention_norm"), vec![2]),
                tensor(name("ffn_norm"), vec![2]),
                tensor(name("ffn_gate"), vec![2, 4]),
                tensor(name("ffn_up"), vec![2, 4]),
                tensor(name("ffn_down"), vec![4, 2]),
                tensor(name("post_ffw_norm"), vec![2]),
                tensor(name("layer_output_scale"), vec![1]),
            ]);
            if layer == 0 {
                tensors.push(tensor(name("attn_v"), vec![2, head]));
            }
        }
        kvs.extend(extra_kvs);
        tensors.extend(extra_tensors);
        build_gguf(&kvs, &tensors, 32)
    }

    /// A `Q8_0` block: scale 0.5 and quants `seed, seed + 1, ...`.
    fn q8_block(seed: u8) -> Vec<u8> {
        let mut block = 0x3800u16.to_le_bytes().to_vec();
        block.extend((0..32).map(|index| seed.wrapping_add(index)));
        block
    }

    /// `load-layered-checkpoint-tensors`: a `gemma4` file's packed
    /// per-layer tables are cut per layer (a `Q8_0` table on its block
    /// boundaries), its fused expert projections split and stacked, and
    /// its router scale folded, through the whole ingest.
    #[test]
    fn slices_per_layer_tables_and_restacks_experts_of_a_gemma4_file() {
        let values = |count: u64| (0..count).map(|i| i as f32 * 0.1 + 0.5).collect::<Vec<_>>();
        let float = |name: &'static str, dimensions: Vec<u64>| TestTensor {
            name,
            data: f32_bytes(&values(dimensions.iter().product())),
            dimensions,
            ggml_type: 0,
        };
        // 4 tokens, 2 layers of 32 per-layer input values, hidden 2;
        // 2 experts of width 2.
        let mut tensors = vec![
            TestTensor {
                name: "per_layer_token_embd.weight",
                dimensions: vec![64, 4],
                ggml_type: 8,
                data: (0..4u8)
                    .flat_map(|token| {
                        (0..2u8).flat_map(move |layer| q8_block(token * 16 + layer * 8))
                    })
                    .collect(),
            },
            float("per_layer_model_proj.weight", vec![2, 64]),
        ];
        for layer in 0..2 {
            let name = |field: &str| -> &'static str {
                Box::leak(format!("blk.{layer}.{field}").into_boxed_str())
            };
            tensors.extend([
                float(name("ffn_gate_up_exps.weight"), vec![2, 4, 2]),
                float(name("ffn_down_exps.weight"), vec![2, 2, 2]),
                float(name("ffn_gate_inp.weight"), vec![2, 2]),
                float(name("ffn_gate_inp.scale"), vec![2]),
                float(name("ffn_down_exps.scale"), vec![2]),
            ]);
        }
        let file = gemma4_gguf(
            32,
            vec![
                kv_uint32("gemma4.expert_count", 2),
                kv_uint32("gemma4.expert_used_count", 1),
                kv_uint32("gemma4.expert_feed_forward_length", 2),
            ],
            tensors,
        );
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(GGUF_FILE_NAME), file).unwrap();
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let result = GgufIngestor::new().ingest(&source).expect("bundle ingests");
        let config = result.manifest.architecture_config.clone().unwrap();
        assert_eq!(config.per_layer_input_size, Some(32));
        let read = |name: &str| {
            let tensor = result
                .manifest
                .tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .unwrap_or_else(|| panic!("no tensor '{name}'"));
            let bytes = result
                .payload_source
                .read_payload(
                    &magnetar_runtime::production_model_ingestion::ProductionPayloadRange {
                        identity: tensor.name.clone(),
                        offset: tensor.offset_bytes.unwrap(),
                        length: tensor.size_bytes.unwrap(),
                        digest: tensor.digest.clone(),
                    },
                )
                .unwrap();
            (tensor.clone(), bytes)
        };
        let as_f32 = |bytes: Vec<u8>| -> Vec<f32> {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect()
        };
        assert!(
            !result
                .manifest
                .tensors
                .iter()
                .any(|tensor| tensor.name.contains("packed."))
        );

        // Layer 1's embedding is each row's second `Q8_0` block, copied.
        let (tensor, bytes) = read("layers.1.per_layer_embedding");
        assert_eq!(tensor.shape, vec![4, 32]);
        assert_eq!(
            tensor.storage_dtype,
            magnetar_runtime::model::ModelDType::Q8
        );
        let expected: Vec<u8> = (0..4u8)
            .flat_map(|token| q8_block(token * 16 + 8))
            .collect();
        assert_eq!(bytes, expected);

        // `[hidden, width]`: element (h, p) is stored row `32 + p`.
        let table = &values(128);
        let (tensor, bytes) = read("layers.1.per_layer_model_projection");
        assert_eq!(tensor.shape, vec![2, 32]);
        let expected: Vec<f32> = (0..2)
            .flat_map(|h| (0..32).map(move |p| table[(32 + p) * 2 + h]))
            .collect();
        assert_eq!(as_f32(bytes), expected);

        // Stacked experts: (h, e * 2 + i) is expert e's gate row i (up row
        // 2 + i); down (i, e * 2 + h) is expert e's output h.
        let gate_up = values(16);
        for (name, first_row) in [("experts_gate_proj", 0), ("experts_up_proj", 2)] {
            let (tensor, bytes) = read(&format!("layers.1.moe.{name}"));
            assert_eq!(tensor.shape, vec![2, 4], "{name}");
            let expected: Vec<f32> = (0..2)
                .flat_map(|h| (0..4).map(move |column| (h, column)))
                .map(|(h, column)| gate_up[((column / 2) * 4 + first_row + column % 2) * 2 + h])
                .collect();
            assert_eq!(as_f32(bytes), expected, "{name}");
        }
        let down = &values(8);
        let (tensor, bytes) = read("layers.1.moe.experts_down_proj");
        assert_eq!(tensor.shape, vec![2, 4]);
        let expected: Vec<f32> = (0..2)
            .flat_map(|i| (0..4).map(move |column| down[column * 2 + i]))
            .collect();
        assert_eq!(as_f32(bytes), expected);

        // The router, `[hidden, experts]`, with `scale[h] / sqrt(hidden)`.
        let router = &values(4);
        let scale = &values(2);
        let (tensor, bytes) = read("layers.1.moe.router_proj");
        assert_eq!(tensor.shape, vec![2, 2]);
        let factor = 2f32.powf(-0.5);
        let expected: Vec<f32> = (0..2)
            .flat_map(|h| (0..2).map(move |e| router[e * 2 + h] * scale[h] * factor))
            .collect();
        assert_eq!(as_f32(bytes), expected);
        let (tensor, bytes) = read("layers.1.moe.per_expert_scale");
        assert_eq!(tensor.shape, vec![2]);
        assert_eq!(as_f32(bytes), values(2));
    }

    /// `per-layer-kv-geometry`: a `gemma4` file ingests with its layers'
    /// geometry, its options, the layered logical names, and no
    /// `rope_freqs` weight.
    #[test]
    fn ingests_a_layered_gemma4_file() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(GGUF_FILE_NAME), tiny_gemma4_gguf()).unwrap();
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let result = GgufIngestor::new().ingest(&source).expect("bundle ingests");
        assert_eq!(result.manifest.architecture.family, "gemma4");
        let config = result.manifest.architecture_config.clone().unwrap();
        assert_eq!(config.head_dim, 2);
        assert_eq!(config.rope_theta, 10_000.0);
        assert_eq!(config.final_logit_softcapping, Some(30.0));
        assert_eq!(config.per_layer_input_size, None);
        let layout = config.layer_layout.unwrap();
        assert_eq!(layout[0].sliding_window, Some(8));
        assert_eq!(layout[0].head_dim, 2);
        assert!(!layout[0].value_from_key);
        assert_eq!(layout[1].sliding_window, None);
        assert_eq!(layout[1].head_dim, 4);
        assert_eq!(layout[1].rope_theta, 1_000_000.0);
        assert_eq!(layout[1].partial_rotary_factor, Some(0.5));
        assert!(layout[1].value_from_key);
        let names: Vec<&str> = result
            .manifest
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect();
        assert!(!names.contains(&"rope_freqs"));
        for name in [
            "layers.1.self_attn.q_norm",
            "layers.1.pre_ffn_norm",
            "layers.1.post_attn_norm",
            "layers.1.layer_scalar",
        ] {
            assert!(names.contains(&name), "{name} in {names:?}");
        }
        // Projections are transposed to `[in, out]`.
        let q = result
            .manifest
            .tensors
            .iter()
            .find(|tensor| tensor.name == "layers.1.self_attn.q_proj")
            .unwrap();
        assert_eq!(q.shape, vec![2, 4]);
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
        // The family is the checkpoint's own architecture name, as for a
        // Hugging Face bundle's model_type.
        assert_eq!(result.manifest.architecture.family, "qwen2");
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

    /// A *quantized*, *non-square* projection weight is never transposed
    /// byte by byte (that would corrupt its block structure,
    /// `resolve-gguf-quantized-projection-transpose-sequencing`): it keeps
    /// its packed bytes and declares transposed storage
    /// (`execute-direct-quantized-weights`), and Model Loading's own
    /// conversion yields the same transposed matrix the byte transpose used
    /// to. Dimensions here (hidden=4, intermediate=8) hold exactly one real
    /// `Q8_0` block (32 elements) for `ffn_gate` and are deliberately
    /// non-square so a transpose bug could not hide behind symmetry.
    /// Ingests a one-layer file whose `ffn_gate` (GGUF ne `[in=4, out=8]`,
    /// one 32-value block) is `gate` stored as `ggml_type`, checks it stays
    /// packed as `dtype` with transposed storage declared, and returns its
    /// values as Model Loading reads them (`[in=4, out=8]`).
    fn ingest_a_quantized_gate(
        ggml_type: u32,
        gate: Vec<u8>,
        dtype: magnetar_runtime::model::ModelDType,
    ) -> Vec<f32> {
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
                ggml_type,
                data: gate,
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
            ffn_gate.storage_dtype, dtype,
            "a quantized tensor keeps its storage dtype"
        );
        assert_eq!(
            ffn_gate.layout.as_deref(),
            Some(magnetar_runtime::packed_weight::TRANSPOSED_STORAGE_LAYOUT)
        );
        // GGUF ne=[in=4, out=8] reversed to HF-native [out=8, in=4], then
        // swapped back to the Component's expected [in=4, out=8].
        assert_eq!(ffn_gate.shape, vec![HIDDEN, INTERMEDIATE]);

        magnetar_runtime::model_loading::host_tensor_from_payload_source(
            ffn_gate,
            result.payload_source.as_ref(),
        )
        .expect("Model Loading converts the packed, transposed-storage tensor")
        .data
    }

    #[test]
    fn keeps_a_non_square_quantized_projection_packed_and_declares_it_transposed() {
        // One real Q8_0 block: d = 1.0, qs = 1..=32.
        let mut gate = 0x3C00u16.to_le_bytes().to_vec(); // d = 1.0
        gate.extend((1i8..=32).map(|value| value as u8));
        let values = ingest_a_quantized_gate(8, gate, magnetar_runtime::model::ModelDType::Q8);
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

    /// A `Q4_0` projection is kept packed with transposed storage declared,
    /// as a `Q8_0` one is, never transposed byte by byte.
    #[test]
    fn keeps_a_q4_0_projection_packed_and_declares_it_transposed() {
        // One Q4_0 block: d = 1.0, value `i` is `i % 16 - 8`, so `qs[j]`
        // holds `j` in both nibbles.
        let mut gate = 0x3C00u16.to_le_bytes().to_vec(); // d = 1.0
        gate.extend((0u8..16).map(|j| j | (j << 4)));
        let values = ingest_a_quantized_gate(2, gate, magnetar_runtime::model::ModelDType::Q4);
        // HF-native [out=8, in=4] value `i` is `i % 16 - 8`; transposed to
        // [in=4, out=8], row `r` holds values `r, r + 4, ..., r + 28`.
        let expected: Vec<f32> = (0..4)
            .flat_map(|r| (0..8).map(move |c| ((r + 4 * c) % 16) as f32 - 8.0))
            .collect();
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
    fn rejects_an_unsupported_architecture_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let kvs = vec![kv_string("general.architecture", "gemma3")];
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

    fn kv_bool(key: &str, value: bool) -> Vec<u8> {
        let mut out = gguf_string(key);
        out.extend(le_u32(7));
        out.push(u8::from(value));
        out
    }

    /// How a `deepseek2` file stores the latent's key and value projection.
    #[derive(Clone, Copy, Debug)]
    enum KeyValueUp {
        /// `attn_kv_b`, as converted before llama.cpp split it.
        Fused,
        /// `attn_k_b` (each head transposed) and `attn_v_b`, in `f32`.
        Split,
        /// The same, with `attn_k_b` in `Q8_0` (`nope` must be 32).
        SplitQ8,
    }

    /// A two-layer `deepseek2` file (hidden 4, 2 heads of `nope` + 2
    /// rotated, values 4 wide, latent 2): a dense first layer, then 2
    /// sigmoid-routed experts of width 2 with a selection bias and one
    /// shared expert. Returns the file and each layer's Hugging Face
    /// `kv_b_proj` (`[heads * (nope + 4), 2]`, row-major), from which the
    /// split form is written as llama.cpp's converter writes it: `k_b =
    /// kv_b.view(heads, nope + v, latent)[:, :nope].transpose(1, 2)` and
    /// `v_b = ...[:, nope:]`.
    fn deepseek2_gguf(nope: u64, form: KeyValueUp) -> (Vec<u8>, Vec<Vec<f32>>) {
        let (heads, rope, value, latent, hidden) = (2u64, 2u64, 4u64, 2u64, 4u64);
        let mut kvs = vec![
            kv_string("general.architecture", "deepseek2"),
            kv_string("general.name", "tiny-deepseek2-test"),
            kv_uint32("deepseek2.embedding_length", hidden as u32),
            kv_uint32("deepseek2.feed_forward_length", 6),
            kv_uint32("deepseek2.block_count", 2),
            kv_uint32("deepseek2.attention.head_count", heads as u32),
            kv_float32("deepseek2.attention.layer_norm_rms_epsilon", 1e-6),
            kv_float32("deepseek2.rope.freq_base", 10_000.0),
            kv_uint32("deepseek2.rope.dimension_count", rope as u32),
            kv_uint32("deepseek2.attention.kv_lora_rank", latent as u32),
            kv_uint32("deepseek2.leading_dense_block_count", 1),
            kv_uint32("deepseek2.expert_count", 2),
            kv_uint32("deepseek2.expert_used_count", 1),
            kv_uint32("deepseek2.expert_feed_forward_length", 2),
            kv_uint32("deepseek2.expert_shared_count", 1),
            kv_float32("deepseek2.expert_weights_scale", 1.5),
            kv_bool("deepseek2.expert_weights_norm", true),
            kv_uint32("deepseek2.expert_gating_func", 2),
            kv_string("deepseek2.rope.scaling.type", "yarn"),
            kv_float32("deepseek2.rope.scaling.factor", 4.0),
            kv_uint32("deepseek2.rope.scaling.original_context_length", 16),
            kv_float32("deepseek2.rope.scaling.yarn_log_multiplier", 0.0707),
            kv_uint32("tokenizer.ggml.bos_token_id", 0),
            kv_uint32("tokenizer.ggml.eos_token_id", 1),
            kv_string("tokenizer.ggml.model", "gpt2"),
            kv_string_array(
                "tokenizer.ggml.tokens",
                &["<bos>", "<eos>", "hello", "world"],
            ),
            kv_string_array("tokenizer.ggml.merges", &[]),
        ];
        match form {
            KeyValueUp::Fused => kvs.extend([
                kv_uint32("deepseek2.attention.head_count_kv", heads as u32),
                kv_uint32("deepseek2.attention.key_length", (nope + rope) as u32),
                kv_uint32("deepseek2.attention.value_length", value as u32),
            ]),
            // The converter's MQA form: one latent "head".
            KeyValueUp::Split | KeyValueUp::SplitQ8 => kvs.extend([
                kv_uint32("deepseek2.attention.head_count_kv", 1),
                kv_uint32("deepseek2.attention.key_length", (latent + rope) as u32),
                kv_uint32("deepseek2.attention.value_length", latent as u32),
                kv_uint32("deepseek2.attention.key_length_mla", (nope + rope) as u32),
                kv_uint32("deepseek2.attention.value_length_mla", value as u32),
            ]),
        }
        let next = std::cell::Cell::new(0u32);
        // Distinct values in every tensor.
        let values = |count: u64| -> Vec<f32> {
            (0..count)
                .map(|_| {
                    next.set(next.get() + 1);
                    next.get() as f32 * 0.25
                })
                .collect()
        };
        let tensor = |name: String, dimensions: Vec<u64>, data: Vec<f32>| TestTensor {
            name: Box::leak(name.into_boxed_str()),
            dimensions,
            ggml_type: 0,
            data: f32_bytes(&data),
        };
        let float = |name: String, dimensions: Vec<u64>| {
            let count = dimensions.iter().product();
            tensor(name, dimensions, values(count))
        };
        // GGUF `ne` order: `[in, out]`, `[in, out, heads or experts]`.
        let mut tensors = vec![
            float("token_embd.weight".into(), vec![hidden, 4]),
            float("output_norm.weight".into(), vec![hidden]),
            float("output.weight".into(), vec![hidden, 4]),
        ];
        let mut kv_b_per_layer = Vec::new();
        for layer in 0..2 {
            let name = |field: &str| format!("blk.{layer}.{field}.weight");
            tensors.extend([
                float(name("attn_norm"), vec![hidden]),
                float(name("attn_q"), vec![hidden, heads * (nope + rope)]),
                float(name("attn_kv_a_mqa"), vec![hidden, latent + rope]),
                float(name("attn_kv_a_norm"), vec![latent]),
                float(name("attn_output"), vec![heads * value, hidden]),
                float(name("ffn_norm"), vec![hidden]),
            ]);
            let kv_b = values(heads * (nope + value) * latent);
            let at = |head: u64, row: u64, column: u64| {
                kv_b[((head * (nope + value) + row) * latent + column) as usize]
            };
            match form {
                KeyValueUp::Fused => tensors.push(tensor(
                    name("attn_kv_b"),
                    vec![latent, heads * (nope + value)],
                    kv_b.clone(),
                )),
                KeyValueUp::Split | KeyValueUp::SplitQ8 => {
                    // `[heads, latent, nope]` and `[heads, value, latent]`.
                    let k_b: Vec<f32> = (0..heads)
                        .flat_map(|head| {
                            (0..latent).flat_map(move |column| {
                                (0..nope).map(move |row| (head, row, column))
                            })
                        })
                        .map(|(head, row, column)| at(head, row, column))
                        .collect();
                    let v_b: Vec<f32> = (0..heads)
                        .flat_map(|head| {
                            (0..value).flat_map(move |row| {
                                (0..latent).map(move |column| (head, row, column))
                            })
                        })
                        .map(|(head, row, column)| at(head, nope + row, column))
                        .collect();
                    let mut keys = tensor(name("attn_k_b"), vec![nope, latent, heads], k_b);
                    if matches!(form, KeyValueUp::SplitQ8) {
                        // One block per row; the expected values are the
                        // blocks' own (`q8_block`), so `kv_b` follows them.
                        keys.ggml_type = 8;
                        keys.data = (0..heads * latent)
                            .flat_map(|row| q8_block(row as u8 * 32))
                            .collect();
                    }
                    tensors.push(keys);
                    tensors.push(tensor(name("attn_v_b"), vec![latent, value, heads], v_b));
                }
            }
            kv_b_per_layer.push(kv_b);
            if layer == 0 {
                tensors.extend([
                    float(name("ffn_gate"), vec![hidden, 6]),
                    float(name("ffn_up"), vec![hidden, 6]),
                    float(name("ffn_down"), vec![6, hidden]),
                ]);
                continue;
            }
            tensors.extend([
                float(name("ffn_gate_inp"), vec![hidden, 2]),
                float(name("ffn_gate_exps"), vec![hidden, 2, 2]),
                float(name("ffn_up_exps"), vec![hidden, 2, 2]),
                float(name("ffn_down_exps"), vec![2, hidden, 2]),
                float(name("ffn_gate_shexp"), vec![hidden, 2]),
                float(name("ffn_up_shexp"), vec![hidden, 2]),
                float(name("ffn_down_shexp"), vec![2, hidden]),
                float(format!("blk.{layer}.exp_probs_b.bias"), vec![2]),
            ]);
        }
        (build_gguf(&kvs, &tensors, 32), kv_b_per_layer)
    }

    fn ingest_bytes(file: Vec<u8>) -> ProductionIngestionResult {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(GGUF_FILE_NAME), file).unwrap();
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let result = GgufIngestor::new().ingest(&source).expect("file ingests");
        // Keep the file alive for the reads.
        std::mem::forget(dir);
        result
    }

    /// Tensor `name`'s logical shape and `f32` values.
    fn read_f32(result: &ProductionIngestionResult, name: &str) -> (Vec<u64>, Vec<f32>) {
        let tensor = result
            .manifest
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap_or_else(|| panic!("no tensor '{name}'"));
        assert_eq!(
            tensor.storage_dtype,
            magnetar_runtime::model::ModelDType::F32,
            "{name}"
        );
        let bytes = result
            .payload_source
            .read_payload(
                &magnetar_runtime::production_model_ingestion::ProductionPayloadRange {
                    identity: tensor.name.clone(),
                    offset: tensor.offset_bytes.unwrap(),
                    length: tensor.size_bytes.unwrap(),
                    digest: tensor.digest.clone(),
                },
            )
            .unwrap();
        let values = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        (tensor.shape.clone(), values)
    }

    /// `express-latent-attention-forms`: a `deepseek2` file declares
    /// latent attention, YaRN, its dense and routed layers and its routing,
    /// and its latent-attention projections reach the Component split, from
    /// either form of the key and value projection.
    #[test]
    fn ingests_a_latent_attention_deepseek2_file_in_either_key_value_form() {
        let (nope, heads, value, latent) = (3u64, 2u64, 4u64, 2u64);
        let mut derived = Vec::new();
        for form in [KeyValueUp::Fused, KeyValueUp::Split] {
            let (file, kv_b) = deepseek2_gguf(nope, form);
            let result = ingest_bytes(file);
            let config = result.manifest.architecture_config.clone().unwrap();
            assert_eq!(
                config.latent_attention,
                Some(magnetar_runtime::model::LatentAttentionConfig {
                    q_lora_rank: None,
                    kv_lora_rank: 2,
                    qk_nope_head_dim: 3,
                    qk_rope_head_dim: 2,
                    v_head_dim: 4,
                    norm_eps: None,
                }),
                "{form:?}"
            );
            assert_eq!((config.head_dim, config.num_key_value_heads), (5, 2));
            assert_eq!(config.rope_scaling_factor, None);
            let Some(magnetar_runtime::model::RopeFrequencyScaling::Yarn {
                scaling,
                mscale_all_dim,
            }) = config.rope_frequency_scaling
            else {
                panic!("no YaRN: {:?}", config.rope_frequency_scaling);
            };
            assert_eq!((scaling.factor, scaling.original_context_length), (4.0, 16));
            assert_eq!((scaling.beta_fast, scaling.beta_slow), (32.0, 1.0));
            assert_eq!((scaling.attention_factor, scaling.truncate), (1.0, true));
            assert!((mscale_all_dim.unwrap() - 0.707).abs() < 1e-6);
            assert_eq!(
                config.mixture_of_experts,
                Some(magnetar_runtime::model::MixtureOfExpertsConfig {
                    experts: 2,
                    experts_per_token: 1,
                    expert_intermediate_size: 2,
                    shared_expert_intermediate_size: Some(2),
                    normalize_top_k: true,
                })
            );
            assert_eq!(config.moe_layers, Some(vec![false, true]));
            assert_eq!(
                config.expert_routing,
                Some(magnetar_runtime::model::ExpertRoutingConfig {
                    scoring: magnetar_runtime::model::ExpertScoring::Sigmoid,
                    groups: None,
                    routed_scaling_factor: 1.5,
                    selection_bias: true,
                    normalization: None,
                    shared_expert_gated: false,
                })
            );
            for (name, shape) in [
                ("self_attn.q_nope_proj", vec![4, 6]),
                ("self_attn.q_rope_proj", vec![4, 4]),
                ("self_attn.kv_a_proj", vec![4, 2]),
                ("self_attn.k_rope_proj", vec![4, 2]),
                ("self_attn.k_nope_proj", vec![2, 6]),
                ("self_attn.v_proj", vec![2, 8]),
                ("self_attn.o_proj", vec![8, 4]),
            ] {
                assert_eq!(
                    read_f32(&result, &format!("layers.1.{name}")).0,
                    shape,
                    "{name}"
                );
            }
            assert_eq!(read_f32(&result, "layers.1.moe.selection_bias").0, vec![2]);
            assert!(
                !result
                    .manifest
                    .tensors
                    .iter()
                    .any(|tensor| tensor.name.contains("packed."))
            );
            // `[latent, heads * width]`: (l, h * width + j) is `kv_b`'s row
            // `h * (nope + value) + offset + j`, column `l`.
            for (layer, kv_b) in kv_b.iter().enumerate() {
                for (name, offset, width) in [("k_nope_proj", 0, nope), ("v_proj", nope, value)] {
                    let expected: Vec<f32> = (0..latent)
                        .flat_map(|l| (0..heads * width).map(move |column| (l, column)))
                        .map(|(l, column)| {
                            let (head, j) = (column / width, column % width);
                            kv_b[((head * (nope + value) + offset + j) * latent + l) as usize]
                        })
                        .collect();
                    let (_, values) =
                        read_f32(&result, &format!("layers.{layer}.self_attn.{name}"));
                    assert_eq!(values, expected, "{form:?} layer {layer} {name}");
                }
                // The absorbed decode's tables: `kv_b`'s key and value rows,
                // `[out, in]` as stored.
                for (name, offset, width) in [("k_absorb", 0, nope), ("v_absorb", nope, value)] {
                    let expected: Vec<f32> = (0..heads * width)
                        .flat_map(|row| {
                            let (head, j) = (row / width, row % width);
                            (0..latent).map(move |l| {
                                kv_b[((head * (nope + value) + offset + j) * latent + l) as usize]
                            })
                        })
                        .collect();
                    let (shape, values) =
                        read_f32(&result, &format!("layers.{layer}.self_attn.{name}"));
                    assert_eq!(shape, vec![heads * width, latent], "{form:?} {name}");
                    assert_eq!(values, expected, "{form:?} layer {layer} {name}");
                }
            }
            derived.push(read_f32(&result, "layers.1.self_attn.q_nope_proj").0);
        }
        assert_eq!(derived[0], derived[1]);
    }

    /// A block-quantized `attn_k_b` reaches the Component dequantized, each
    /// head transposed back.
    #[test]
    fn a_quantized_split_key_projection_is_transposed_back_in_f32() {
        let (nope, heads, latent) = (32u64, 2u64, 2u64);
        let (file, _) = deepseek2_gguf(nope, KeyValueUp::SplitQ8);
        let result = ingest_bytes(file);
        // Stored row `h * latent + l` is `q8_block((h * latent + l) * 32)`:
        // 0.5 times its quants.
        let stored =
            |head: u64, l: u64, j: u64| 0.5 * f32::from(((head * latent + l) * 32 + j) as u8 as i8);
        let expected: Vec<f32> = (0..latent)
            .flat_map(|l| (0..heads * nope).map(move |column| (l, column)))
            .map(|(l, column)| stored(column / nope, l, column % nope))
            .collect();
        for layer in 0..2 {
            let (shape, values) =
                read_f32(&result, &format!("layers.{layer}.self_attn.k_nope_proj"));
            assert_eq!(shape, vec![latent, heads * nope]);
            assert_eq!(values, expected, "layer {layer}");
            // The absorbed key table: the same values, kept `[out, in]`.
            let (shape, values) = read_f32(&result, &format!("layers.{layer}.self_attn.k_absorb"));
            assert_eq!(shape, vec![heads * nope, latent]);
            let kept: Vec<f32> = (0..heads * nope)
                .flat_map(|row| (0..latent).map(move |l| stored(row / nope, l, row % nope)))
                .collect();
            assert_eq!(values, kept, "layer {layer}");
        }
    }
}
