//! Real GGUF tensor discovery and bounded payload access
//! (`magnetar-format-gguf`'s parser composed, not reimplemented -- mirrors
//! `loaders/huggingface::weights`'s own discipline for Safetensors).
//! GGUF stores every tensor's raw bytes in the *same single file*, unlike
//! Hugging Face's separate/sharded Safetensors files, so there is no
//! shard discovery step here -- only one file to resolve and parse.
//!
//! Quantization is out of scope for this ingestor (a separate,
//! not-yet-implemented Magnetar chantier: dequantization-at-load or
//! quantized compute kernels do not exist yet) -- any tensor GGUF
//! declares with a quantized `ggml_type` (`Q4_K`/`Q5_K`/`Q8_0`) is
//! rejected structurally, not silently accepted and later failed deep
//! inside weight materialization.

use magnetar_runtime::model::ModelTensorMetadata;
use magnetar_runtime::production_model_ingestion::{
    ProductionArtifactPayloadSource, ProductionIngestionError, ProductionModelSource,
    ProductionPayloadRange,
};
use std::collections::BTreeMap;
use std::{
    fs,
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
};

const GGUF_FILE_NAME: &str = "model.gguf";

#[derive(Debug)]
struct TensorLocation {
    offset: u64,
    length: u64,
}

/// Bounded, on-demand [`ProductionArtifactPayloadSource`] for a GGUF file:
/// `read_payload` opens the file, seeks to `tensor_data_start + offset`,
/// and reads exactly that many bytes -- it never holds the whole file's
/// bytes in memory across calls, matching
/// `loaders/huggingface::SafetensorsPayloadSource`'s identical discipline.
#[derive(Debug)]
pub struct GgufPayloadSource {
    path: PathBuf,
    tensor_data_start: u64,
    locations: BTreeMap<String, TensorLocation>,
}

impl ProductionArtifactPayloadSource for GgufPayloadSource {
    fn read_payload(
        &self,
        range: &ProductionPayloadRange,
    ) -> Result<Vec<u8>, ProductionIngestionError> {
        let location = self.locations.get(&range.identity).ok_or_else(|| {
            ProductionIngestionError::PayloadOutOfBounds {
                identity: range.identity.clone(),
            }
        })?;
        if range.offset != location.offset || range.length != location.length {
            return Err(ProductionIngestionError::PayloadOutOfBounds {
                identity: range.identity.clone(),
            });
        }
        let mut file = fs::File::open(&self.path).map_err(|error| {
            ProductionIngestionError::PayloadUnavailable {
                identity: format!("{}: {error}", range.identity),
            }
        })?;
        let start = self
            .tensor_data_start
            .checked_add(location.offset)
            .ok_or_else(|| ProductionIngestionError::PayloadOutOfBounds {
                identity: range.identity.clone(),
            })?;
        file.seek(SeekFrom::Start(start)).map_err(|error| {
            ProductionIngestionError::PayloadUnavailable {
                identity: format!("{}: {error}", range.identity),
            }
        })?;
        let length = usize::try_from(location.length).map_err(|_| {
            ProductionIngestionError::PayloadOutOfBounds {
                identity: range.identity.clone(),
            }
        })?;
        let mut buffer = vec![0u8; length];
        file.read_exact(&mut buffer).map_err(|error| {
            ProductionIngestionError::PayloadUnavailable {
                identity: format!("{}: {error}", range.identity),
            }
        })?;
        if let Some(expected_digest) = &range.digest {
            expected_digest.verify_bytes(&buffer).map_err(|error| {
                ProductionIngestionError::IntegrityMismatch {
                    identity: format!("{}: {error}", range.identity),
                }
            })?;
        }
        Ok(buffer)
    }
}

/// [`discover_and_parse_weights`]'s success value: the normalized
/// (renamed, shape-reversed) tensor inventory, a bounded payload source,
/// and the file's raw key-value metadata for architecture/tokenizer
/// normalization.
type DiscoveredGgufWeights = (
    Vec<ModelTensorMetadata>,
    GgufPayloadSource,
    BTreeMap<String, magnetar_format_gguf::GgufMetadataValue>,
);

/// Discovers and parses this bundle's single `model.gguf` file.
pub fn discover_and_parse_weights(
    source: &ProductionModelSource,
) -> Result<DiscoveredGgufWeights, ProductionIngestionError> {
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
    drop(bytes);

    let mut locations = BTreeMap::new();
    let mut tensors = Vec::with_capacity(artifact.tensors.len());
    for mut tensor in artifact.tensors {
        if tensor.quantization.is_some() {
            return Err(ProductionIngestionError::UnsupportedFormat {
                reason: format!(
                    "tensor '{}' is quantized ({:?}); GGUF quantization support is a separate, \
                     not-yet-implemented chantier (no dequantization-at-load or quantized \
                     compute kernels exist yet) -- only unquantized F32/F16/BF16 GGUF tensors \
                     are supported today",
                    tensor.name, tensor.quantization
                ),
            });
        }
        let (offset, length) = match (tensor.offset_bytes, tensor.size_bytes) {
            (Some(offset), Some(length)) => (offset, length),
            _ => {
                return Err(ProductionIngestionError::MalformedMetadata {
                    reason: format!("tensor '{}' has no declared byte range", tensor.name),
                });
            }
        };
        // GGUF's `ne[]` dimension order is the reverse of PyTorch/
        // Safetensors row-major shape order (GGML's fastest-varying
        // dimension is `ne[0]`, matching PyTorch's *last* dimension for a
        // standard C-contiguous tensor) -- the underlying byte buffer for
        // a row-major tensor is identical either way, only the shape
        // *labeling* needs reversing to match the convention every other
        // Magnetar tensor consumer (the Qwen Component, Reference CPU/
        // CUDA kernels) expects. A 1-D tensor's reversal is a no-op.
        tensor.shape.reverse();

        let canonical = crate::naming::normalize_tensor_name(&tensor.name)?;
        locations.insert(canonical.clone(), TensorLocation { offset, length });
        tensor.name = canonical;
        tensors.push(tensor);
    }

    Ok((
        tensors,
        GgufPayloadSource {
            path,
            tensor_data_start: artifact.tensor_data_start,
            locations,
        },
        artifact.metadata,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnetar_runtime::ModelArtifactSource;

    /// Builds a minimal, real, valid GGUF byte blob for tests -- mirrors
    /// `magnetar-format-gguf`'s own test-only `build_gguf` helper (a
    /// separate crate, so it cannot be reused directly): magic, version,
    /// key-value metadata, tensor-info section, alignment padding, then
    /// raw tensor data.
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
        out.extend(le_u32(8)); // GGUF_METADATA_VALUE_TYPE_STRING
        out.extend(gguf_string(value));
        out
    }

    struct TestTensor {
        name: &'static str,
        dimensions: Vec<u64>,
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
            tensor_info_bytes.extend(le_u32(0)); // ggml_type: F32
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

    #[test]
    fn discovers_and_reads_a_gguf_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = build_gguf(
            &[kv_string("general.name", "test-model")],
            &[
                TestTensor {
                    name: "token_embd.weight",
                    dimensions: vec![4, 2], // GGUF ne order: [in, out] = [4, 2]
                    data: vec![0u8; 4 * 2 * 4],
                },
                TestTensor {
                    name: "blk.0.attn_q.weight",
                    dimensions: vec![2, 2],
                    data: {
                        let mut bytes = Vec::new();
                        for value in [1.0f32, 2.0, 3.0, 4.0] {
                            bytes.extend_from_slice(&value.to_le_bytes());
                        }
                        bytes
                    },
                },
            ],
            32,
        );
        fs::write(dir.path().join(GGUF_FILE_NAME), &file).unwrap();

        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let (tensors, payload_source, metadata) = discover_and_parse_weights(&source).unwrap();
        assert_eq!(tensors.len(), 2);
        assert_eq!(
            metadata.get("general.name"),
            Some(&magnetar_format_gguf::GgufMetadataValue::String(
                "test-model".into()
            ))
        );

        let embedding = tensors
            .iter()
            .find(|t| t.name == "token_embedding")
            .unwrap();
        // GGUF's ne order [4, 2] must be reversed to PyTorch row-major [2, 4].
        assert_eq!(embedding.shape, vec![2, 4]);

        let q_proj = tensors
            .iter()
            .find(|t| t.name == "layers.0.self_attn.q_proj")
            .unwrap();
        let range = ProductionPayloadRange {
            identity: q_proj.name.clone(),
            offset: q_proj.offset_bytes.unwrap(),
            length: q_proj.size_bytes.unwrap(),
            digest: None,
        };
        let bytes = payload_source.read_payload(&range).unwrap();
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rejects_missing_gguf_file() {
        let dir = tempfile::tempdir().unwrap();
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let error = discover_and_parse_weights(&source).unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::RequiredPartMissing { .. }
        ));
    }
}
