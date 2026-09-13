//! Real GGUF tensor discovery and bounded payload access
//! (`magnetar-format-gguf`'s parser composed, not reimplemented -- mirrors
//! `loaders/huggingface::weights`'s own discipline for Safetensors).
//! GGUF stores every tensor's raw bytes in the *same single file*, unlike
//! Hugging Face's separate/sharded Safetensors files, so there is no
//! shard discovery step here -- only one file to resolve and parse.
//!
//! A quantized tensor (`Q8_0`/`Q4_K`/`Q5_K`) is dequantized to `F32` right
//! here, at payload-read time, *before* `weight_layout.rs`'s projection
//! transpose ever sees it (`resolve-gguf-quantized-projection-transpose-sequencing`):
//! that transpose assumes a flat per-element byte width, which is
//! meaningless for block-quantized data, so dequantizing first and
//! presenting the result as plain `F32` (overriding the discovered
//! tensor's declared `storage_dtype`/`size_bytes`/`quantization`
//! accordingly) means every later step in this crate's pipeline treats a
//! formerly-quantized tensor exactly like a real `F32` one, with no
//! special-casing needed downstream. `magnetar-runtime`'s own generic
//! `Q8_0`/`Q4_K`/`Q5_K` dequantization
//! (`support-gguf-quantized-tensor-dequantization`) is consequently never
//! reached for a GGUF-sourced tensor -- this crate fully resolves to `F32`
//! before Model Loading ever sees it.

use crate::dequantize;
use magnetar_runtime::model::{ModelDType, ModelTensorMetadata};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorSource {
    Plain,
    Q8_0,
    Q4K,
    Q5K,
}

/// One tensor's location in the original GGUF file, and how to turn its
/// raw bytes into what Model Loading actually requests. `file_offset`/
/// `file_length` describe the *original* bytes as declared by
/// `magnetar-format-gguf` (quantized or not); `declared_length` is what a
/// caller's `ProductionPayloadRange.length` must equal -- the same as
/// `file_length` for [`TensorSource::Plain`], or the dequantized `F32`
/// byte count otherwise (matching the overridden `size_bytes` this
/// crate's [`discover_and_parse_weights`] puts on the returned
/// `ModelTensorMetadata`).
#[derive(Debug)]
struct TensorLocation {
    file_offset: u64,
    file_length: u64,
    declared_length: u64,
    source: TensorSource,
}

fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Bounded, on-demand [`ProductionArtifactPayloadSource`] for a GGUF file:
/// `read_payload` opens the file, seeks to `tensor_data_start + offset`,
/// reads exactly the *original* bytes, and (for a quantized tensor)
/// dequantizes them to `F32` before returning -- it never holds the whole
/// file's bytes in memory across calls, matching
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
        if range.offset != location.file_offset || range.length != location.declared_length {
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
            .checked_add(location.file_offset)
            .ok_or_else(|| ProductionIngestionError::PayloadOutOfBounds {
                identity: range.identity.clone(),
            })?;
        file.seek(SeekFrom::Start(start)).map_err(|error| {
            ProductionIngestionError::PayloadUnavailable {
                identity: format!("{}: {error}", range.identity),
            }
        })?;
        let read_length = usize::try_from(location.file_length).map_err(|_| {
            ProductionIngestionError::PayloadOutOfBounds {
                identity: range.identity.clone(),
            }
        })?;
        let mut raw = vec![0u8; read_length];
        file.read_exact(&mut raw).map_err(|error| {
            ProductionIngestionError::PayloadUnavailable {
                identity: format!("{}: {error}", range.identity),
            }
        })?;
        // A declared digest (if any) describes the *original* file bytes,
        // exactly like `loaders/huggingface::weight_layout`'s identical
        // "verify before transform" discipline for its own byte
        // transpose -- checked here, before dequantization, never against
        // the converted representation.
        if let Some(expected_digest) = &range.digest {
            expected_digest.verify_bytes(&raw).map_err(|error| {
                ProductionIngestionError::IntegrityMismatch {
                    identity: format!("{}: {error}", range.identity),
                }
            })?;
        }
        Ok(match location.source {
            TensorSource::Plain => raw,
            TensorSource::Q8_0 => f32_le_bytes(&dequantize::dequantize_q8_0(&raw)),
            TensorSource::Q4K => f32_le_bytes(&dequantize::dequantize_q4_k(&raw)),
            TensorSource::Q5K => f32_le_bytes(&dequantize::dequantize_q5_k(&raw)),
        })
    }
}

fn element_count(shape: &[u64]) -> Result<u64, ProductionIngestionError> {
    shape
        .iter()
        .try_fold(1u64, |count, &dimension| count.checked_mul(dimension))
        .ok_or_else(|| ProductionIngestionError::MalformedMetadata {
            reason: "tensor element count overflowed".into(),
        })
}

/// [`discover_and_parse_weights`]'s success value: the normalized
/// (renamed, shape-reversed, dequantized-where-applicable) tensor
/// inventory, a bounded payload source, and the file's raw key-value
/// metadata for architecture/tokenizer normalization.
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
        let (file_offset, file_length) = match (tensor.offset_bytes, tensor.size_bytes) {
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
        // CUDA kernels) expects. A 1-D tensor's reversal is a no-op. This
        // is a pure logical-shape relabeling, independent of quantization
        // -- element counts are unaffected either way.
        tensor.shape.reverse();

        let (source_kind, declared_length) = match tensor.storage_dtype {
            ModelDType::Q8 | ModelDType::Q4K | ModelDType::Q5K => {
                let elements = element_count(&tensor.shape)?;
                let block_elements = match tensor.storage_dtype {
                    ModelDType::Q8 => dequantize::Q8_0_BLOCK_ELEMENTS,
                    ModelDType::Q4K | ModelDType::Q5K => dequantize::QK_BLOCK_ELEMENTS,
                    _ => unreachable!(),
                };
                if !elements.is_multiple_of(block_elements) {
                    return Err(ProductionIngestionError::MalformedMetadata {
                        reason: format!(
                            "tensor '{}' element count {elements} is not a multiple of its \
                             quantization format's block size {block_elements}",
                            tensor.name
                        ),
                    });
                }
                let dequantized_bytes = elements.checked_mul(4).ok_or_else(|| {
                    ProductionIngestionError::MalformedMetadata {
                        reason: format!(
                            "tensor '{}' dequantized byte size overflowed",
                            tensor.name
                        ),
                    }
                })?;
                let source_kind = match tensor.storage_dtype {
                    ModelDType::Q8 => TensorSource::Q8_0,
                    ModelDType::Q4K => TensorSource::Q4K,
                    ModelDType::Q5K => TensorSource::Q5K,
                    _ => unreachable!(),
                };
                // Fully resolved to F32 here: `weight_layout`'s transpose
                // and every later step never learns this tensor was ever
                // quantized.
                tensor.storage_dtype = ModelDType::F32;
                tensor.size_bytes = Some(dequantized_bytes);
                tensor.quantization = None;
                tensor.digest = None;
                (source_kind, dequantized_bytes)
            }
            _ => (TensorSource::Plain, file_length),
        };

        let canonical = crate::naming::normalize_tensor_name(&tensor.name)?;
        locations.insert(
            canonical.clone(),
            TensorLocation {
                file_offset,
                file_length,
                declared_length,
                source: source_kind,
            },
        );
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

    #[test]
    fn discovers_and_reads_a_gguf_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = build_gguf(
            &[kv_string("general.name", "test-model")],
            &[
                TestTensor {
                    name: "token_embd.weight",
                    dimensions: vec![4, 2], // GGUF ne order: [in, out] = [4, 2]
                    ggml_type: 0,
                    data: vec![0u8; 4 * 2 * 4],
                },
                TestTensor {
                    name: "blk.0.attn_q.weight",
                    dimensions: vec![2, 2],
                    ggml_type: 0,
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

    /// GGUF `ggml_type` 8 = `Q8_0`. A quantized tensor's declared
    /// `storage_dtype`/`size_bytes`/`quantization` must all be overridden
    /// to reflect the *dequantized* `F32` reality, and reading it must
    /// return the real dequantized bytes -- not the original quantized
    /// ones, and not a rejection.
    #[test]
    fn dequantizes_a_q8_0_tensor_and_overrides_its_declared_dtype() {
        let dir = tempfile::tempdir().unwrap();
        let mut q8_data = Vec::with_capacity(34);
        q8_data.extend_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
        q8_data.extend((1i8..=32).map(|value| value as u8)); // qs = 1..=32
        let file = build_gguf(
            &[],
            &[TestTensor {
                name: "blk.0.attn_q.weight",
                dimensions: vec![32], // 1-D so ne-reversal is a no-op
                ggml_type: 8,         // GGML_TYPE_Q8_0
                data: q8_data,
            }],
            32,
        );
        fs::write(dir.path().join(GGUF_FILE_NAME), &file).unwrap();

        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let (tensors, payload_source, _metadata) = discover_and_parse_weights(&source).unwrap();
        let tensor = tensors
            .iter()
            .find(|t| t.name == "layers.0.self_attn.q_proj")
            .unwrap();
        assert_eq!(tensor.storage_dtype, ModelDType::F32);
        assert_eq!(tensor.size_bytes, Some(32 * 4));
        assert!(tensor.quantization.is_none());

        let range = ProductionPayloadRange {
            identity: tensor.name.clone(),
            offset: tensor.offset_bytes.unwrap(),
            length: tensor.size_bytes.unwrap(),
            digest: None,
        };
        let bytes = payload_source.read_payload(&range).unwrap();
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        let expected: Vec<f32> = (1..=32).map(|value| value as f32).collect();
        assert_eq!(values, expected);
    }
}
