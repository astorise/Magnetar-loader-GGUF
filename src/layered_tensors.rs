//! A `gemma4` file's packed tensors, cut into the tensors its Model
//! Component requests (`load-layered-checkpoint-tensors`), as
//! `loaders/huggingface`'s module of the same name does for a Hugging Face
//! bundle:
//! - `per_layer_token_embd` (`[vocab, layers * width]`): layer `N` reads
//!   columns `[N * width, (N + 1) * width)` as `layers.N.per_layer_embedding`;
//! - `per_layer_model_proj` (`[layers * width, hidden]`): layer `N` reads the
//!   matching rows as `layers.N.per_layer_model_projection`;
//! - `blk.N.ffn_gate_up_exps` (`[experts, 2 * intermediate, hidden]`, each
//!   expert's gate rows before its up rows, as the converter copies
//!   `experts.gate_up_proj`) splits into the stacked `moe.experts_gate_proj`
//!   and `moe.experts_up_proj` (`[experts * intermediate, hidden]`);
//! - `blk.N.ffn_gate_inp` with `blk.N.ffn_gate_inp.scale` and `hidden^-0.5`
//!   folded into its input columns becomes `moe.router_proj`, in `f32`. The
//!   converter copies `router.scale` unchanged, and the router computes
//!   `proj(norm(x) * scale * hidden^-0.5)`.
//!
//! A `deepseek2` file's latent-attention projections are split by rows
//! (`express-latent-attention-forms`), as the Hugging Face loader splits
//! the checkpoint's: `attn_q` (or `attn_q_b`) into `self_attn.q_nope_proj`
//! and `self_attn.q_rope_proj`, each head's position-independent rows before
//! its rotated rows; `attn_kv_a_mqa` into `self_attn.kv_a_proj` and
//! `self_attn.k_rope_proj`, the latent's rows before the shared rotated
//! key's; and `attn_kv_b` into `self_attn.k_nope_proj` and
//! `self_attn.v_proj`, each head's key rows before its value rows. Files
//! converted after llama.cpp split `attn_kv_b` store `attn_v_b`
//! (`[heads, value, latent]`, the value rows already apart: the same bytes)
//! and `attn_k_b` (`[heads, latent, key]`, each head's key rows
//! transposed): its heads are transposed back, in the stored float type,
//! or in `f32` from a block-quantized one, whose blocks run along the wrong
//! dimension to be kept.
//!
//! Slices are taken in stored bytes: every cut falls on a row or, for a
//! block-quantized table, on a block boundary, so block-quantized bytes are
//! copied, never re-encoded. The derived tensors keep the `[out, in]`
//! orientation, so `weight_layout.rs` orients them like every stored
//! projection.

use crate::quant_blocks;
use crate::weights::GgufPayloadSource;
use magnetar_runtime::model::{ModelArchitectureConfig, ModelDType, ModelTensorMetadata};
use magnetar_runtime::production_model_ingestion::{
    ProductionArtifactPayloadSource, ProductionIngestionError, ProductionPayloadRange,
};
use std::collections::BTreeMap;

/// `per_layer_token_embd`, as discovered.
pub(crate) const PER_LAYER_EMBEDDING_TABLE: &str = "packed.per_layer_token_embd";
/// `per_layer_model_proj`, as discovered.
pub(crate) const PER_LAYER_PROJECTION_TABLE: &str = "packed.per_layer_model_proj";
/// A layer's `ffn_gate_up_exps`, as discovered (after `layers.N.`).
pub(crate) const EXPERTS_GATE_UP: &str = "packed.experts_gate_up";
/// A layer's `ffn_gate_inp`, as discovered (after `layers.N.`).
pub(crate) const ROUTER_PROJECTION: &str = "packed.router_proj";
/// A layer's `ffn_gate_inp.scale`, as discovered (after `layers.N.`).
pub(crate) const ROUTER_SCALE: &str = "packed.router_scale";
/// A latent-attention layer's `attn_q_b`, as discovered (after `layers.N.`).
pub(crate) const QUERY_UP_PROJECTION: &str = "packed.q_b_proj";
/// A latent-attention layer's `attn_kv_a_mqa`, as discovered.
pub(crate) const LATENT_DOWN_PROJECTION: &str = "packed.kv_a_proj_with_mqa";
/// A latent-attention layer's `attn_kv_b`, as discovered.
pub(crate) const LATENT_UP_PROJECTION: &str = "packed.kv_b_proj";
/// A latent-attention layer's `attn_k_b`, as discovered.
pub(crate) const KEY_UP_PER_HEAD: &str = "packed.k_b_proj";
/// A latent-attention layer's `attn_v_b`, as discovered.
pub(crate) const VALUE_UP_PER_HEAD: &str = "packed.v_b_proj";

/// How one derived tensor's bytes come from a stored one.
#[derive(Debug)]
enum Derivation {
    /// `width` bytes at `start` of each of `rows` rows of `row_bytes`.
    Columns {
        source: String,
        rows: u64,
        row_bytes: u64,
        start: u64,
        width: u64,
    },
    /// These `(offset, length)` byte ranges, concatenated.
    Ranges {
        source: String,
        ranges: Vec<(u64, u64)>,
    },
    /// The `[heads, rows, columns]` source with each head transposed, as the
    /// `[heads * columns, rows]` matrix: in `dtype` when it is a float type,
    /// in `f32` otherwise.
    HeadTranspose {
        source: String,
        dtype: ModelDType,
        heads: u64,
        rows: u64,
        columns: u64,
    },
    /// The `[experts, hidden]` router projection with `scale[h] * factor`
    /// folded into its column `h`, as `f32`.
    Router {
        projection: String,
        projection_dtype: ModelDType,
        scale: String,
        scale_dtype: ModelDType,
        hidden: u64,
        factor: f32,
    },
}

/// Answers each derived tensor's identity from the stored tensors, and
/// every other identity from `inner`.
#[derive(Debug)]
pub struct LayeredTensorSource {
    inner: GgufPayloadSource,
    derived: BTreeMap<String, (Derivation, u64)>,
}

fn malformed(reason: String) -> ProductionIngestionError {
    ProductionIngestionError::MalformedMetadata { reason }
}

fn is_float(dtype: ModelDType) -> bool {
    matches!(dtype, ModelDType::F32 | ModelDType::F16 | ModelDType::Bf16)
}

/// The stored bytes of one `row_elements`-wide row of `tensor`, which holds
/// `rows` rows; a block-quantized row must hold whole blocks of
/// `granule_elements` (a column slice's width, or the row itself).
fn row_bytes(
    tensor: &ModelTensorMetadata,
    rows: u64,
    row_elements: u64,
    granule_elements: u64,
) -> Result<u64, ProductionIngestionError> {
    let unsupported = |why: &str| ProductionIngestionError::UnsupportedFormat {
        reason: format!("tensor '{}' {why}", tensor.name),
    };
    match quant_blocks::block_elements(tensor.storage_dtype) {
        Some(block) if !granule_elements.is_multiple_of(block) => {
            return Err(unsupported(&format!(
                "would be cut inside a {block}-element quantization block"
            )));
        }
        Some(_) => {}
        None if is_float(tensor.storage_dtype) => {}
        None => {
            return Err(unsupported(&format!(
                "is stored as {:?}, which slicing or restacking does not support",
                tensor.storage_dtype
            )));
        }
    }
    let total = tensor
        .size_bytes
        .ok_or_else(|| malformed(format!("tensor '{}' has no byte size", tensor.name)))?;
    if rows == 0 || row_elements == 0 || !total.is_multiple_of(rows) {
        return Err(malformed(format!(
            "tensor '{}' of {total} bytes is not {rows} rows",
            tensor.name
        )));
    }
    Ok(total / rows)
}

fn decode(bytes: &[u8], dtype: ModelDType) -> Vec<f32> {
    match dtype {
        ModelDType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect(),
        ModelDType::Bf16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| f32::from_bits(u32::from(u16::from_le_bytes(*chunk)) << 16))
            .collect(),
        _ => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| f16_to_f32(u16::from_le_bytes(*chunk)))
            .collect(),
    }
}

/// IEEE 754 binary16 to binary32, subnormals included.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        31 if mantissa == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

/// Removes and returns the tensor named `name`.
fn take(tensors: &mut Vec<ModelTensorMetadata>, name: &str) -> Option<ModelTensorMetadata> {
    let index = tensors.iter().position(|tensor| tensor.name == name)?;
    Some(tensors.remove(index))
}

/// The layer indices of the tensors named `layers.N.{suffix}`.
fn layers_with(tensors: &[ModelTensorMetadata], suffix: &str) -> Vec<String> {
    tensors
        .iter()
        .filter_map(|tensor| {
            let (layer, rest) = tensor.name.strip_prefix("layers.")?.split_once('.')?;
            (rest == suffix).then(|| layer.to_string())
        })
        .collect()
}

/// A derived tensor's metadata: `source`'s, renamed and reshaped, with no
/// declared digest (its bytes are not the file's).
fn derived_tensor(
    source: &ModelTensorMetadata,
    name: String,
    shape: Vec<u64>,
    storage_dtype: ModelDType,
    size_bytes: u64,
) -> ModelTensorMetadata {
    let mut tensor = source.clone();
    tensor.name = name;
    tensor.shape = shape;
    tensor.storage_dtype = storage_dtype;
    tensor.size_bytes = Some(size_bytes);
    tensor.digest = None;
    tensor.layout = None;
    tensor
}

fn matrix(tensor: &ModelTensorMetadata) -> Result<[u64; 2], ProductionIngestionError> {
    match tensor.shape[..] {
        [rows, columns] => Ok([rows, columns]),
        _ => Err(malformed(format!(
            "'{}' is not a matrix: {:?}",
            tensor.name, tensor.shape
        ))),
    }
}

/// Replaces the packed tensors in `tensors` with the tensors derived from
/// them, and wraps `inner` to read the derived ones. A file without packed
/// tensors is returned unchanged.
pub fn derive(
    tensors: &mut Vec<ModelTensorMetadata>,
    inner: GgufPayloadSource,
    config: &ModelArchitectureConfig,
) -> Result<LayeredTensorSource, ProductionIngestionError> {
    let mut derived = BTreeMap::new();
    let mut added = Vec::new();
    let layers = u64::from(config.num_hidden_layers);

    let tables = [
        take(tensors, PER_LAYER_EMBEDDING_TABLE),
        take(tensors, PER_LAYER_PROJECTION_TABLE),
    ];
    if tables.iter().any(Option::is_some) {
        let width = u64::from(config.per_layer_input_size.ok_or_else(|| {
            malformed(
                "the file packs per-layer input tables, but its metadata declares no \
                 embedding_length_per_layer_input"
                    .into(),
            )
        })?);
        let [embedding, projection] = tables;
        if let Some(table) = embedding {
            let [vocab, packed] = matrix(&table)?;
            if packed != layers * width {
                return Err(malformed(format!(
                    "'{}' has {packed} columns, not {layers} layers of {width}",
                    table.name
                )));
            }
            let stored_row = row_bytes(&table, vocab, packed, width)?;
            let slice = stored_row / layers;
            for layer in 0..layers {
                let name = format!("layers.{layer}.per_layer_embedding");
                derived.insert(
                    name.clone(),
                    (
                        Derivation::Columns {
                            source: table.name.clone(),
                            rows: vocab,
                            row_bytes: stored_row,
                            start: layer * slice,
                            width: slice,
                        },
                        vocab * slice,
                    ),
                );
                added.push(derived_tensor(
                    &table,
                    name,
                    vec![vocab, width],
                    table.storage_dtype,
                    vocab * slice,
                ));
            }
        }
        if let Some(table) = projection {
            let [packed, hidden] = matrix(&table)?;
            if packed != layers * width {
                return Err(malformed(format!(
                    "'{}' has {packed} rows, not {layers} layers of {width}",
                    table.name
                )));
            }
            let size = row_bytes(&table, packed, hidden, hidden)? * width;
            for layer in 0..layers {
                let name = format!("layers.{layer}.per_layer_model_projection");
                derived.insert(
                    name.clone(),
                    (
                        Derivation::Ranges {
                            source: table.name.clone(),
                            ranges: vec![(layer * size, size)],
                        },
                        size,
                    ),
                );
                added.push(derived_tensor(
                    &table,
                    name,
                    vec![width, hidden],
                    table.storage_dtype,
                    size,
                ));
            }
        }
    }

    for layer in layers_with(tensors, EXPERTS_GATE_UP) {
        let stored = take(tensors, &format!("layers.{layer}.{EXPERTS_GATE_UP}"))
            .expect("listed by layers_with");
        let [experts, gate_up, hidden] = stored.shape[..] else {
            return Err(malformed(format!(
                "'{}' is not [experts, 2 * intermediate, hidden]: {:?}",
                stored.name, stored.shape
            )));
        };
        if !gate_up.is_multiple_of(2) {
            return Err(malformed(format!(
                "'{}' has an odd gate and up width {gate_up}",
                stored.name
            )));
        }
        let intermediate = gate_up / 2;
        if let Some(moe) = &config.mixture_of_experts
            && (experts != u64::from(moe.experts) || intermediate != moe.expert_intermediate_size)
        {
            return Err(malformed(format!(
                "'{}' holds {experts} experts of width {intermediate}, but the metadata \
                 declares {} of width {}",
                stored.name, moe.experts, moe.expert_intermediate_size
            )));
        }
        let half = row_bytes(&stored, experts * gate_up, hidden, hidden)? * intermediate;
        for (projection, first) in [("experts_gate_proj", 0), ("experts_up_proj", half)] {
            let name = format!("layers.{layer}.moe.{projection}");
            derived.insert(
                name.clone(),
                (
                    Derivation::Ranges {
                        source: stored.name.clone(),
                        ranges: (0..experts)
                            .map(|expert| (expert * 2 * half + first, half))
                            .collect(),
                    },
                    experts * half,
                ),
            );
            added.push(derived_tensor(
                &stored,
                name,
                vec![experts * intermediate, hidden],
                stored.storage_dtype,
                experts * half,
            ));
        }
    }

    for layer in layers_with(tensors, ROUTER_PROJECTION) {
        let projection = take(tensors, &format!("layers.{layer}.{ROUTER_PROJECTION}"))
            .expect("listed by layers_with");
        let scale = take(tensors, &format!("layers.{layer}.{ROUTER_SCALE}")).ok_or_else(|| {
            malformed(format!(
                "layer {layer}'s router has a projection but no scale"
            ))
        })?;
        let [experts, hidden] = matrix(&projection)?;
        if scale.shape != [hidden] {
            return Err(malformed(format!(
                "'{}' is {:?}, not [{hidden}]",
                scale.name, scale.shape
            )));
        }
        for tensor in [&projection, &scale] {
            if !is_float(tensor.storage_dtype) {
                return Err(ProductionIngestionError::UnsupportedFormat {
                    reason: format!(
                        "router tensor '{}' is stored as {:?}; folding its scale needs a \
                         float router",
                        tensor.name, tensor.storage_dtype
                    ),
                });
            }
        }
        let name = format!("layers.{layer}.moe.router_proj");
        let size = experts * hidden * 4;
        derived.insert(
            name.clone(),
            (
                Derivation::Router {
                    projection: projection.name.clone(),
                    projection_dtype: projection.storage_dtype,
                    scale: scale.name.clone(),
                    scale_dtype: scale.storage_dtype,
                    hidden,
                    factor: (hidden as f32).powf(-0.5),
                },
                size,
            ),
        );
        added.push(derived_tensor(
            &projection,
            name,
            vec![experts, hidden],
            ModelDType::F32,
            size,
        ));
    }
    derive_latent_attention(tensors, config, &mut derived, &mut added)?;
    if let Some(stray) = tensors
        .iter()
        .find(|tensor| tensor.name.contains("packed."))
    {
        return Err(malformed(format!(
            "tensor '{}' has no matching tensor to derive weights from",
            stray.name
        )));
    }
    tensors.extend(added);
    Ok(LayeredTensorSource { inner, derived })
}

/// The bytes of one element of a float `dtype`.
fn float_bytes(dtype: ModelDType) -> Option<usize> {
    match dtype {
        ModelDType::F32 => Some(4),
        ModelDType::F16 | ModelDType::Bf16 => Some(2),
        _ => None,
    }
}

/// `raw`, stored as block-quantized `dtype`, as `f32` values.
fn dequantize(raw: &[u8], dtype: ModelDType) -> Option<Vec<f32>> {
    use magnetar_runtime::quantization as q;
    Some(match dtype {
        ModelDType::Q8 => q::dequantize_q8_0(raw),
        ModelDType::Q4 => q::dequantize_q4_0(raw),
        ModelDType::Q4K => q::dequantize_q4_k(raw),
        ModelDType::Q5K => q::dequantize_q5_k(raw),
        ModelDType::Q6K => q::dequantize_q6_k(raw),
        ModelDType::Q2K => q::dequantize_q2_k(raw),
        ModelDType::Q3K => q::dequantize_q3_k(raw),
        ModelDType::Iq4Nl => q::dequantize_iq4_nl(raw),
        ModelDType::Iq4Xs => q::dequantize_iq4_xs(raw),
        ModelDType::Iq2Xxs => q::dequantize_iq2_xxs(raw),
        ModelDType::Iq2Xs => q::dequantize_iq2_xs(raw),
        ModelDType::Iq2S => q::dequantize_iq2_s(raw),
        ModelDType::Iq3Xxs => q::dequantize_iq3_xxs(raw),
        ModelDType::Iq3S => q::dequantize_iq3_s(raw),
        _ => return None,
    })
}

/// `rows` of `stored`, by `(first row, row count)`, as the derived tensor
/// `layers.{layer}.{name}`: a `[rows, in]` matrix of `stored`'s bytes.
fn insert_rows(
    derived: &mut BTreeMap<String, (Derivation, u64)>,
    added: &mut Vec<ModelTensorMetadata>,
    stored: &ModelTensorMetadata,
    layer: &str,
    name: &str,
    rows: &[(u64, u64)],
) -> Result<(), ProductionIngestionError> {
    let [stored_rows, columns] = matrix(stored)?;
    let row = row_bytes(stored, stored_rows, columns, columns)?;
    let count: u64 = rows.iter().map(|(_, count)| count).sum();
    let name = format!("layers.{layer}.{name}");
    derived.insert(
        name.clone(),
        (
            Derivation::Ranges {
                source: stored.name.clone(),
                ranges: rows
                    .iter()
                    .map(|(first, count)| (first * row, count * row))
                    .collect(),
            },
            count * row,
        ),
    );
    added.push(derived_tensor(
        stored,
        name,
        vec![count, columns],
        stored.storage_dtype,
        count * row,
    ));
    Ok(())
}

/// Each head's `first` rows then its `second` rows, for `heads` heads: the
/// `(first row, row count)` runs of either part.
fn per_head_rows(heads: u64, first: u64, second: u64) -> [Vec<(u64, u64)>; 2] {
    let stride = first + second;
    [
        (0..heads).map(|head| (head * stride, first)).collect(),
        (0..heads)
            .map(|head| (head * stride + first, second))
            .collect(),
    ]
}

/// Takes `layers.{layer}.{suffix}`, checked to be `shape`.
fn take_shaped(
    tensors: &mut Vec<ModelTensorMetadata>,
    layer: &str,
    suffix: &str,
    shape: &[u64],
) -> Result<ModelTensorMetadata, ProductionIngestionError> {
    let name = format!("layers.{layer}.{suffix}");
    let stored = take(tensors, &name)
        .ok_or_else(|| malformed(format!("latent-attention layer {layer} has no '{name}'")))?;
    if stored.shape != shape {
        return Err(malformed(format!(
            "'{name}' is {:?}, not {shape:?}",
            stored.shape
        )));
    }
    Ok(stored)
}

/// Splits each latent-attention layer's fused projections by rows, and
/// transposes a split key projection's heads back (see the module
/// documentation).
fn derive_latent_attention(
    tensors: &mut Vec<ModelTensorMetadata>,
    config: &ModelArchitectureConfig,
    derived: &mut BTreeMap<String, (Derivation, u64)>,
    added: &mut Vec<ModelTensorMetadata>,
) -> Result<(), ProductionIngestionError> {
    let Some(mla) = &config.latent_attention else {
        if let Some(layer) = layers_with(tensors, LATENT_DOWN_PROJECTION).first() {
            return Err(malformed(format!(
                "layer {layer} stores latent-attention projections, but the metadata \
                 declares no kv_lora_rank"
            )));
        }
        return Ok(());
    };
    let heads = u64::from(config.num_attention_heads);
    let (nope, rope, value) = (mla.qk_nope_head_dim, mla.qk_rope_head_dim, mla.v_head_dim);
    let latent = u64::from(mla.kv_lora_rank);
    let hidden = config.hidden_size;
    for layer in 0..u64::from(config.num_hidden_layers) {
        let layer = layer.to_string();
        let (query_name, query_input) = match mla.q_lora_rank {
            Some(rank) => (QUERY_UP_PROJECTION, u64::from(rank)),
            None => ("self_attn.q_proj", hidden),
        };
        let query = take_shaped(
            tensors,
            &layer,
            query_name,
            &[heads * (nope + rope), query_input],
        )?;
        let [query_nope, query_rope] = per_head_rows(heads, nope, rope);
        insert_rows(
            derived,
            added,
            &query,
            &layer,
            "self_attn.q_nope_proj",
            &query_nope,
        )?;
        insert_rows(
            derived,
            added,
            &query,
            &layer,
            "self_attn.q_rope_proj",
            &query_rope,
        )?;

        let down = take_shaped(
            tensors,
            &layer,
            LATENT_DOWN_PROJECTION,
            &[latent + rope, hidden],
        )?;
        insert_rows(
            derived,
            added,
            &down,
            &layer,
            "self_attn.kv_a_proj",
            &[(0, latent)],
        )?;
        insert_rows(
            derived,
            added,
            &down,
            &layer,
            "self_attn.k_rope_proj",
            &[(latent, rope)],
        )?;

        if tensors
            .iter()
            .any(|tensor| tensor.name == format!("layers.{layer}.{LATENT_UP_PROJECTION}"))
        {
            let up = take_shaped(
                tensors,
                &layer,
                LATENT_UP_PROJECTION,
                &[heads * (nope + value), latent],
            )?;
            let [key_nope, values] = per_head_rows(heads, nope, value);
            insert_rows(
                derived,
                added,
                &up,
                &layer,
                "self_attn.k_nope_proj",
                &key_nope,
            )?;
            insert_rows(derived, added, &up, &layer, "self_attn.v_proj", &values)?;
            continue;
        }
        // `[heads, value, latent]`: already the `[heads * value, latent]`
        // value projection's bytes.
        let mut values = take_shaped(tensors, &layer, VALUE_UP_PER_HEAD, &[heads, value, latent])?;
        values.shape = vec![heads * value, latent];
        insert_rows(
            derived,
            added,
            &values,
            &layer,
            "self_attn.v_proj",
            &[(0, heads * value)],
        )?;
        let keys = take_shaped(tensors, &layer, KEY_UP_PER_HEAD, &[heads, latent, nope])?;
        if float_bytes(keys.storage_dtype).is_none()
            && dequantize(&[], keys.storage_dtype).is_none()
        {
            return Err(ProductionIngestionError::UnsupportedFormat {
                reason: format!(
                    "tensor '{}' is stored as {:?}, which transposing does not support",
                    keys.name, keys.storage_dtype
                ),
            });
        }
        let dtype = if float_bytes(keys.storage_dtype).is_some() {
            keys.storage_dtype
        } else {
            ModelDType::F32
        };
        let name = format!("layers.{layer}.self_attn.k_nope_proj");
        let size = heads * nope * latent * float_bytes(dtype).expect("a float type") as u64;
        derived.insert(
            name.clone(),
            (
                Derivation::HeadTranspose {
                    source: keys.name.clone(),
                    dtype: keys.storage_dtype,
                    heads,
                    rows: latent,
                    columns: nope,
                },
                size,
            ),
        );
        added.push(derived_tensor(
            &keys,
            name,
            vec![heads * nope, latent],
            dtype,
            size,
        ));
    }
    Ok(())
}

impl ProductionArtifactPayloadSource for LayeredTensorSource {
    fn read_payload(
        &self,
        range: &ProductionPayloadRange,
    ) -> Result<Vec<u8>, ProductionIngestionError> {
        let Some((derivation, length)) = self.derived.get(&range.identity) else {
            return self.inner.read_payload(range);
        };
        if range.length != *length {
            return Err(ProductionIngestionError::PayloadOutOfBounds {
                identity: range.identity.clone(),
            });
        }
        let mut bytes = Vec::with_capacity(usize::try_from(*length).unwrap_or(0));
        match derivation {
            Derivation::Columns {
                source,
                rows,
                row_bytes,
                start,
                width,
            } => {
                bytes = self
                    .inner
                    .read_rows(source, *start, *row_bytes, *width, *rows)?;
            }
            Derivation::Ranges { source, ranges } => {
                for (offset, length) in ranges {
                    bytes.extend(self.inner.read_within(source, *offset, *length)?);
                }
            }
            Derivation::HeadTranspose {
                source,
                dtype,
                heads,
                rows,
                columns,
            } => {
                let total = self.inner.tensor_length(source)?;
                let raw = self.inner.read_within(source, 0, total)?;
                let (heads, rows, columns) = (*heads as usize, *rows as usize, *columns as usize);
                // Element `(head, row, column)` of the source goes to row
                // `head * columns + column`, column `row`.
                let source_index =
                    |head: usize, column: usize, row: usize| (head * rows + row) * columns + column;
                let order = (0..heads).flat_map(|head| {
                    (0..columns)
                        .flat_map(move |column| (0..rows).map(move |row| (head, column, row)))
                });
                match float_bytes(*dtype) {
                    Some(size) => {
                        for (head, column, row) in order {
                            let at = source_index(head, column, row) * size;
                            bytes.extend_from_slice(&raw[at..at + size]);
                        }
                    }
                    None => {
                        let values = dequantize(&raw, *dtype).ok_or_else(|| {
                            malformed(format!("'{source}' cannot be dequantized"))
                        })?;
                        if values.len() != heads * rows * columns {
                            return Err(malformed(format!(
                                "'{source}' dequantized to {} values, not {}",
                                values.len(),
                                heads * rows * columns
                            )));
                        }
                        for (head, column, row) in order {
                            bytes.extend_from_slice(
                                &values[source_index(head, column, row)].to_le_bytes(),
                            );
                        }
                    }
                }
            }
            Derivation::Router {
                projection,
                projection_dtype,
                scale,
                scale_dtype,
                hidden,
                factor,
            } => {
                let read_all = |identity: &str| {
                    let total = self.inner.tensor_length(identity)?;
                    self.inner.read_within(identity, 0, total)
                };
                let weights = decode(&read_all(projection)?, *projection_dtype);
                let scale = decode(&read_all(scale)?, *scale_dtype);
                for (index, weight) in weights.iter().enumerate() {
                    let folded = weight * scale[index % *hidden as usize] * factor;
                    bytes.extend_from_slice(&folded.to_le_bytes());
                }
            }
        }
        if bytes.len() as u64 != *length {
            return Err(malformed(format!(
                "'{}' derived {} bytes, not {length}",
                range.identity,
                bytes.len()
            )));
        }
        Ok(bytes)
    }
}
