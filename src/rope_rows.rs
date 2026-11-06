//! Restores the query and key projection rows llama.cpp's converter
//! permuted for its interleaved-pair RoPE (`LlamaModel.permute` in
//! `convert_hf_to_gguf.py`) to the checkpoint's own row order, so a GGUF
//! file reaches the Model Component exactly as its upstream checkpoint
//! stores it and the Component's half-split RoPE applies unchanged.
//!
//! Within each head of `head_dim` rows, the converter moves upstream row
//! `half * head_dim / 2 + i` to row `2 * i + half`. Restoring it moves
//! whole rows -- `[out_features, in_features]` row-major, so a row is
//! `in_features` contiguous values, or a whole number of quantization
//! blocks for a block-quantized projection -- and never changes a value.

use magnetar_runtime::model::ModelTensorMetadata;
use magnetar_runtime::production_model_ingestion::{
    ProductionArtifactPayloadSource, ProductionIngestionError, ProductionPayloadRange,
};
use std::collections::BTreeMap;

/// Whether `canonical_name` names a projection the converter permuted.
fn is_permuted_projection(canonical_name: &str) -> bool {
    canonical_name.ends_with("self_attn.q_proj") || canonical_name.ends_with("self_attn.k_proj")
}

/// The upstream position of each stored row of a permuted projection of
/// `rows` rows in heads of `head_dim` rows: `restored[upstream] =
/// stored[source_row(upstream)]`.
fn source_row(upstream: u64, head_dim: u64) -> u64 {
    let head = upstream / head_dim;
    let within = upstream % head_dim;
    let half = within / (head_dim / 2);
    let i = within % (head_dim / 2);
    head * head_dim + 2 * i + half
}

/// Wraps a payload source, restoring the row order of every permuted
/// projection named in `rows` (its `[out_features]` row count, as stored)
/// and passing every other identity through unchanged.
pub struct RopeRowRestoringPayloadSource<S> {
    inner: S,
    head_dim: u64,
    rows: BTreeMap<String, u64>,
}

impl<S> RopeRowRestoringPayloadSource<S> {
    /// Builds the wrapper from `tensors` in stored `[out_features,
    /// in_features]` order (before any shape swap), and clears the
    /// declared digest of each permuted projection: it described the
    /// permuted bytes, which this source never returns. With `permuted`
    /// false (an architecture the converter does not permute) it passes
    /// every tensor through.
    pub fn new(
        inner: S,
        tensors: &mut [ModelTensorMetadata],
        permuted: bool,
        head_dim: u64,
    ) -> Result<Self, ProductionIngestionError> {
        let mut rows = BTreeMap::new();
        for tensor in tensors.iter_mut() {
            if !permuted || !is_permuted_projection(&tensor.name) {
                continue;
            }
            let [out_features, _] = tensor.shape[..] else {
                return Err(ProductionIngestionError::MalformedMetadata {
                    reason: format!("'{}' is not a 2D projection", tensor.name),
                });
            };
            if head_dim < 2 || !head_dim.is_multiple_of(2) || !out_features.is_multiple_of(head_dim)
            {
                return Err(ProductionIngestionError::MalformedMetadata {
                    reason: format!(
                        "'{}' has {out_features} rows, not whole heads of {head_dim} rows with \
                         rotated pairs",
                        tensor.name
                    ),
                });
            }
            tensor.digest = None;
            rows.insert(tensor.name.clone(), out_features);
        }
        Ok(Self {
            inner,
            head_dim,
            rows,
        })
    }
}

impl<S: ProductionArtifactPayloadSource> ProductionArtifactPayloadSource
    for RopeRowRestoringPayloadSource<S>
{
    fn read_payload(
        &self,
        range: &ProductionPayloadRange,
    ) -> Result<Vec<u8>, ProductionIngestionError> {
        let bytes = self.inner.read_payload(range)?;
        let Some(&rows) = self.rows.get(&range.identity) else {
            return Ok(bytes);
        };
        if !(bytes.len() as u64).is_multiple_of(rows) {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: format!(
                    "'{}' holds {} bytes, not {rows} equal rows",
                    range.identity,
                    bytes.len()
                ),
            });
        }
        let row_bytes = bytes.len() / rows as usize;
        let mut restored = vec![0u8; bytes.len()];
        for upstream in 0..rows {
            let source = source_row(upstream, self.head_dim) as usize;
            let destination = upstream as usize;
            restored[destination * row_bytes..(destination + 1) * row_bytes]
                .copy_from_slice(&bytes[source * row_bytes..(source + 1) * row_bytes]);
        }
        Ok(restored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// llama.cpp's `permute`: `w.reshape(n_head, 2, rows / n_head / 2,
    /// cols).swapaxes(1, 2).reshape(rows, cols)`, on row indices.
    fn llama_cpp_permute(rows: u64, head_dim: u64) -> Vec<u64> {
        let mut stored = Vec::new();
        for head in 0..rows / head_dim {
            for i in 0..head_dim / 2 {
                for half in 0..2 {
                    stored.push(head * head_dim + half * (head_dim / 2) + i);
                }
            }
        }
        stored
    }

    #[test]
    fn restores_the_upstream_row_of_every_permuted_row() {
        let (rows, head_dim) = (24, 8);
        // stored[r] = upstream row the converter put at r.
        let stored = llama_cpp_permute(rows, head_dim);
        for (stored_row, &upstream) in stored.iter().enumerate() {
            assert_eq!(source_row(upstream, head_dim), stored_row as u64);
        }
    }

    struct Fixed(Vec<u8>);

    impl ProductionArtifactPayloadSource for Fixed {
        fn read_payload(
            &self,
            _range: &ProductionPayloadRange,
        ) -> Result<Vec<u8>, ProductionIngestionError> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn moves_whole_rows_of_any_width_and_passes_other_tensors_through() {
        let (rows, head_dim, row_bytes) = (8_u64, 4_u64, 3_usize);
        let stored = llama_cpp_permute(rows, head_dim);
        // Each stored row's bytes name the upstream row it holds.
        let bytes: Vec<u8> = stored
            .iter()
            .flat_map(|&upstream| std::iter::repeat_n(upstream as u8, row_bytes))
            .collect();
        let tensor = |name: &str| ModelTensorMetadata {
            name: name.into(),
            shape: vec![rows, 2],
            storage_dtype: magnetar_runtime::model::ModelDType::Q8,
            layout: None,
            shard: None,
            offset_bytes: Some(0),
            size_bytes: Some(bytes.len() as u64),
            quantization: None,
            expected_compute_dtype: None,
            digest: Some(magnetar_runtime::ModelDigest::sha256(&bytes)),
        };
        let mut tensors = vec![
            tensor("layers.0.self_attn.k_proj"),
            tensor("layers.0.self_attn.v_proj"),
        ];
        let source =
            RopeRowRestoringPayloadSource::new(Fixed(bytes.clone()), &mut tensors, true, head_dim)
                .unwrap();
        assert!(tensors[0].digest.is_none());
        assert!(tensors[1].digest.is_some());
        let read = |identity: &str| {
            source
                .read_payload(&ProductionPayloadRange {
                    identity: identity.into(),
                    offset: 0,
                    length: bytes.len() as u64,
                    digest: None,
                })
                .unwrap()
        };
        let restored: Vec<u8> = (0..rows as u8)
            .flat_map(|upstream| std::iter::repeat_n(upstream, row_bytes))
            .collect();
        assert_eq!(read("layers.0.self_attn.k_proj"), restored);
        assert_eq!(read("layers.0.self_attn.v_proj"), bytes);
    }
}
