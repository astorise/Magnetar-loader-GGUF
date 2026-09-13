//! Projection weight transposition -- identical requirement and algorithm
//! to `loaders/huggingface::weight_layout` (that crate cannot be depended
//! on from here; both externalized Loader modules independently need
//! this). GGUF's own `ne[]` dimension-order convention describes the
//! *same, untransposed* raw bytes a Hugging Face Safetensors file would
//! (llama.cpp's real GGUF writer applies no Q/K permutation and no
//! general matrix transpose for Qwen2 -- see this ingestor's governing
//! design.md for the sourced research backing that claim): after
//! `weights.rs` reverses GGUF's `ne[]`-order shape into the Hugging
//! Face-equivalent `[out_features, in_features]` labeling, the tensor
//! bytes underneath are byte-for-byte what a Safetensors file would have
//! stored for the same checkpoint. The Runtime's Qwen Component expects
//! every projection's *logical* shape as `[in_features, out_features]`
//! (`y = x @ W` directly, not `nn.Linear`'s `y = x @ W.T`) -- this module
//! performs the exact same real byte-level transpose
//! `loaders/huggingface` performs for the same reason.

use magnetar_runtime::model::ModelTensorMetadata;
use magnetar_runtime::production_model_ingestion::{
    ProductionArtifactPayloadSource, ProductionIngestionError, ProductionPayloadRange,
};
use std::collections::BTreeMap;

/// Whether `canonical_name` (already normalized -- see `naming.rs`) names
/// a projection weight this Runtime expects transposed relative to how a
/// Hugging Face-equivalent layout stores it (and, since no byte-level
/// permutation happens in GGUF conversion, how GGUF's own raw bytes are
/// still laid out too). Matches `lm_head` exactly and every `*_proj` name
/// via the shared `proj` substring -- deliberately not `token_embedding`
/// (a lookup table) or the 1D normalization vectors.
fn is_transposed_projection(canonical_name: &str) -> bool {
    canonical_name == "lm_head" || canonical_name.contains("proj")
}

/// Swaps every 2D projection tensor's declared shape from
/// `[out_features, in_features]` to `[in_features, out_features]`, in
/// place, and clears its declared content digest (the digest, if any, was
/// computed over the pre-transpose bytes; `TransposingPayloadSource`
/// verifies it against those same pre-transpose bytes at read time, at
/// the correct boundary, so a stale post-transpose comparison elsewhere
/// must not run).
pub fn swap_declared_projection_shapes(tensors: &mut [ModelTensorMetadata]) {
    for tensor in tensors.iter_mut() {
        if is_transposed_projection(&tensor.name) && tensor.shape.len() == 2 {
            tensor.shape.swap(0, 1);
            tensor.digest = None;
        }
    }
}

struct TransposedTensorShape {
    out_features: u64,
    in_features: u64,
    element_bytes: u64,
}

/// Wraps an inner payload source, transposing the bytes returned for
/// every projection tensor identity in `original_shapes` (row-major
/// `[out_features, in_features]` -> `[in_features, out_features]`,
/// preserving each element's own byte width) and passing every other
/// identity through unchanged.
pub struct TransposingPayloadSource<S> {
    inner: S,
    original_shapes: BTreeMap<String, TransposedTensorShape>,
}

impl<S> TransposingPayloadSource<S> {
    /// Builds the wrapper from `tensors`' state *before*
    /// [`swap_declared_projection_shapes`] is applied to them.
    pub fn new(inner: S, tensors: &[ModelTensorMetadata]) -> Self {
        let original_shapes = tensors
            .iter()
            .filter(|tensor| is_transposed_projection(&tensor.name) && tensor.shape.len() == 2)
            .map(|tensor| {
                (
                    tensor.name.clone(),
                    TransposedTensorShape {
                        out_features: tensor.shape[0],
                        in_features: tensor.shape[1],
                        element_bytes: tensor.storage_dtype.descriptor().size_bytes(),
                    },
                )
            })
            .collect();
        Self {
            inner,
            original_shapes,
        }
    }
}

impl<S: ProductionArtifactPayloadSource> ProductionArtifactPayloadSource
    for TransposingPayloadSource<S>
{
    fn read_payload(
        &self,
        range: &ProductionPayloadRange,
    ) -> Result<Vec<u8>, ProductionIngestionError> {
        let bytes = self.inner.read_payload(range)?;
        let Some(shape) = self.original_shapes.get(&range.identity) else {
            return Ok(bytes);
        };
        let element_count = shape
            .out_features
            .checked_mul(shape.in_features)
            .ok_or_else(|| ProductionIngestionError::MalformedMetadata {
                reason: format!("'{}' element count overflows", range.identity),
            })?;
        if bytes.len() as u64 != element_count.saturating_mul(shape.element_bytes) {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: format!(
                    "'{}' byte length {} does not match {}x{} elements at {} bytes/element",
                    range.identity,
                    bytes.len(),
                    shape.out_features,
                    shape.in_features,
                    shape.element_bytes
                ),
            });
        }
        let rows = shape.out_features as usize;
        let cols = shape.in_features as usize;
        let element_bytes = shape.element_bytes as usize;
        let mut transposed = vec![0u8; bytes.len()];
        for row in 0..rows {
            for col in 0..cols {
                let source_offset = (row * cols + col) * element_bytes;
                let dest_offset = (col * rows + row) * element_bytes;
                transposed[dest_offset..dest_offset + element_bytes]
                    .copy_from_slice(&bytes[source_offset..source_offset + element_bytes]);
            }
        }
        Ok(transposed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnetar_runtime::model::ModelDType;
    use std::collections::BTreeMap as StdBTreeMap;

    struct FixedPayloadSource(StdBTreeMap<String, Vec<u8>>);

    impl ProductionArtifactPayloadSource for FixedPayloadSource {
        fn read_payload(
            &self,
            range: &ProductionPayloadRange,
        ) -> Result<Vec<u8>, ProductionIngestionError> {
            self.0.get(&range.identity).cloned().ok_or_else(|| {
                ProductionIngestionError::PayloadOutOfBounds {
                    identity: range.identity.clone(),
                }
            })
        }
    }

    fn tensor(name: &str, shape: Vec<u64>, dtype: ModelDType) -> ModelTensorMetadata {
        ModelTensorMetadata {
            name: name.to_string(),
            shape,
            storage_dtype: dtype,
            layout: None,
            shard: None,
            offset_bytes: Some(0),
            size_bytes: Some(0),
            quantization: None,
            expected_compute_dtype: None,
            digest: None,
        }
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn swaps_projection_shapes_but_not_embedding_or_norms() {
        let mut tensors = vec![
            tensor("layers.0.self_attn.q_proj", vec![10, 4], ModelDType::F32),
            tensor("token_embedding", vec![100, 4], ModelDType::F32),
            tensor("layers.0.input_norm", vec![4], ModelDType::F32),
        ];
        swap_declared_projection_shapes(&mut tensors);
        assert_eq!(tensors[0].shape, vec![4, 10]);
        assert_eq!(tensors[1].shape, vec![100, 4]);
        assert_eq!(tensors[2].shape, vec![4]);
    }

    #[test]
    fn transposes_a_non_square_projection_correctly() {
        // 2x3 (out=2, in=3): [[1,2,3],[4,5,6]]
        let raw = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut bytes_by_name = StdBTreeMap::new();
        bytes_by_name.insert("layers.0.mlp.gate_proj".to_string(), raw);
        let tensors = vec![tensor(
            "layers.0.mlp.gate_proj",
            vec![2, 3],
            ModelDType::F32,
        )];
        let source = TransposingPayloadSource::new(FixedPayloadSource(bytes_by_name), &tensors);
        let transposed = source
            .read_payload(&ProductionPayloadRange {
                identity: "layers.0.mlp.gate_proj".into(),
                offset: 0,
                length: 24,
                digest: None,
            })
            .unwrap();
        let values: Vec<f32> = transposed
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        // 3x2 transpose: [[1,4],[2,5],[3,6]]
        assert_eq!(values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
