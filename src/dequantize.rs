//! Real dequantization for GGUF's `Q8_0`/`Q4_K`/`Q5_K` block formats --
//! identical algorithm to `magnetar-runtime`'s own `model_loading.rs`
//! dequantization (that crate cannot be depended on for this; both
//! independently port the same real upstream `ggml-org/llama.cpp`
//! `ggml-quants.c` formulas, verified directly against the real source,
//! not recalled from memory).
//!
//! This crate needs its own copy specifically so it can dequantize
//! *before* applying its projection-weight transpose
//! (`weight_layout.rs`): that transpose operates on raw bytes assuming a
//! flat per-element width, which is meaningless for block-quantized data.
//! Dequantizing here first means the transpose only ever sees plain `F32`
//! bytes, exactly like it already does for a real `F32`/`F16`/`BF16` GGUF
//! tensor -- `magnetar-runtime`'s own generic `Q8_0`/`Q4_K`/`Q5_K` support
//! (`support-gguf-quantized-tensor-dequantization`) is not reached at all
//! for a tensor this crate has already resolved to `F32` before Model
//! Loading ever sees it.

/// GGUF `block_q8_0`: 32 elements, 34 bytes (`ggml_half d` then 32 signed
/// `int8` quants), no padding.
pub const Q8_0_BLOCK_ELEMENTS: u64 = 32;
pub const Q8_0_BLOCK_BYTES: u64 = 34;
/// GGUF K-quant super-block element count (`QK_K` upstream): 8 sub-blocks
/// of 32 elements each, shared by `block_q4_K` and `block_q5_K`.
pub const QK_BLOCK_ELEMENTS: u64 = 256;
/// `block_q4_K`: `2 (d) + 2 (dmin) + 12 (scales) + 128 (qs) = 144` bytes.
pub const Q4_K_BLOCK_BYTES: u64 = 144;
/// `block_q5_K`: `2 (d) + 2 (dmin) + 12 (scales) + 32 (qh) + 128 (qs) = 176` bytes.
pub const Q5_K_BLOCK_BYTES: u64 = 176;
const K_SCALE_SIZE: usize = 12;

/// Converts one IEEE 754 binary16 ("half float") value to `f32`, exactly
/// -- verbatim copy of `magnetar-runtime`'s own already-tested
/// `f16_to_f32` (`f16_to_f32_handles_every_numeric_class_exactly` there
/// covers every numeric class explicitly: `pub(crate)` there, so this
/// crate cannot import it, matching the "ported, not shared" convention
/// this file's own module doc already documents). A GGUF `ggml_half`
/// scale is a genuine IEEE-754 binary16, not a bespoke float type,
/// confirmed against real upstream `ggml-impl.h`.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = u32::from(bits & 0x3FF);
    let magnitude_bits = if exponent == 0 {
        if mantissa == 0 {
            0
        } else {
            // Subnormal `f16`: renormalize by shifting the mantissa left
            // until its implicit leading bit would land at position 10,
            // counting how many shifts that took to compute the correct
            // (negative, then rebiased) `f32` exponent.
            let mut mantissa = mantissa;
            let mut shift = 0u32;
            while mantissa & 0x400 == 0 {
                mantissa <<= 1;
                shift += 1;
            }
            mantissa &= 0x3FF;
            let f32_exponent = 127 - 15 - shift + 1;
            (f32_exponent << 23) | (mantissa << 13)
        }
    } else if exponent == 0x1F {
        // Infinity (mantissa == 0) or NaN (mantissa != 0): `f32`'s
        // all-ones exponent field means the same thing.
        (0xFFu32 << 23) | (mantissa << 13)
    } else {
        // `exponent` (1..=30) minus f16's bias (15) can be negative before
        // rebiasing into f32's own (127) -- must go through signed
        // arithmetic, unlike the always-non-negative subnormal branch
        // above, or a small-but-normal f16 exponent (e.g. 14, for `0.5`)
        // underflows this as `u32` subtraction.
        let f32_exponent = (i32::from(exponent) - 15 + 127) as u32;
        (f32_exponent << 23) | (mantissa << 13)
    };
    f32::from_bits(sign | magnitude_bits)
}

/// Unpacks one K-quant sub-block's 6-bit scale and 6-bit min from the
/// shared 12-byte packed `scales` field -- ports `ggml-quants.c`'s
/// `get_scale_min_k4` bit-for-bit.
fn get_scale_min_k4(j: usize, scales: &[u8; K_SCALE_SIZE]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

/// Dequantizes one `block_q8_0` region (`raw.len()` a multiple of 34
/// bytes): `value[i] = d * qs[i]`.
pub fn dequantize_q8_0(raw: &[u8]) -> Vec<f32> {
    raw.as_chunks::<{ Q8_0_BLOCK_BYTES as usize }>()
        .0
        .iter()
        .flat_map(|block| {
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            block[2..].iter().map(move |&q| d * f32::from(q as i8))
        })
        .collect()
}

/// Dequantizes one `block_q4_K` region (`raw.len()` a multiple of 144
/// bytes), porting `ggml-quants.c`'s `dequantize_row_q4_K` bit-for-bit.
pub fn dequantize_q4_k(raw: &[u8]) -> Vec<f32> {
    let mut out =
        Vec::with_capacity(raw.len() / Q4_K_BLOCK_BYTES as usize * QK_BLOCK_ELEMENTS as usize);
    for block in raw.as_chunks::<{ Q4_K_BLOCK_BYTES as usize }>().0 {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales: [u8; K_SCALE_SIZE] = block[4..4 + K_SCALE_SIZE].try_into().unwrap();
        let qs = &block[4 + K_SCALE_SIZE..4 + K_SCALE_SIZE + 128];

        let mut is = 0usize;
        for chunk_start in (0..QK_BLOCK_ELEMENTS as usize).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, &scales);
            let (d1, min1) = (d * f32::from(sc1), dmin * f32::from(m1));
            let (sc2, m2) = get_scale_min_k4(is + 1, &scales);
            let (d2, min2) = (d * f32::from(sc2), dmin * f32::from(m2));
            let q = &qs[chunk_start / 2..chunk_start / 2 + 32];
            out.extend(q.iter().map(|&byte| d1 * f32::from(byte & 0xF) - min1));
            out.extend(q.iter().map(|&byte| d2 * f32::from(byte >> 4) - min2));
            is += 2;
        }
    }
    out
}

/// Dequantizes one `block_q5_K` region (`raw.len()` a multiple of 176
/// bytes), porting `ggml-quants.c`'s `dequantize_row_q5_K` bit-for-bit.
pub fn dequantize_q5_k(raw: &[u8]) -> Vec<f32> {
    let mut out =
        Vec::with_capacity(raw.len() / Q5_K_BLOCK_BYTES as usize * QK_BLOCK_ELEMENTS as usize);
    for block in raw.as_chunks::<{ Q5_K_BLOCK_BYTES as usize }>().0 {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales: [u8; K_SCALE_SIZE] = block[4..4 + K_SCALE_SIZE].try_into().unwrap();
        let qh = &block[4 + K_SCALE_SIZE..4 + K_SCALE_SIZE + 32];
        let qs = &block[4 + K_SCALE_SIZE + 32..4 + K_SCALE_SIZE + 32 + 128];

        let mut is = 0usize;
        let (mut u1, mut u2) = (1u8, 2u8);
        for chunk_start in (0..QK_BLOCK_ELEMENTS as usize).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, &scales);
            let (d1, min1) = (d * f32::from(sc1), dmin * f32::from(m1));
            let (sc2, m2) = get_scale_min_k4(is + 1, &scales);
            let (d2, min2) = (d * f32::from(sc2), dmin * f32::from(m2));
            let ql = &qs[chunk_start / 2..chunk_start / 2 + 32];
            out.extend(ql.iter().zip(qh).map(|(&byte, &high)| {
                let value = (byte & 0xF) | if high & u1 != 0 { 16 } else { 0 };
                d1 * f32::from(value) - min1
            }));
            out.extend(ql.iter().zip(qh).map(|(&byte, &high)| {
                let value = (byte >> 4) | if high & u2 != 0 { 16 } else { 0 };
                d2 * f32::from(value) - min2
            }));
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same edge-case coverage as `magnetar-runtime`'s
    /// `f16_to_f32_handles_every_numeric_class_exactly`, proving this
    /// ported copy is exact, not just "close enough" for typical scale
    /// values.
    #[test]
    fn f16_to_f32_handles_every_numeric_class_exactly() {
        assert_eq!(f16_to_f32(0x0000).to_bits(), 0f32.to_bits());
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0f32).to_bits());
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x03FF), 2f32.powi(-14) * (1023.0 / 1024.0));
        assert_eq!(f16_to_f32(0x7BFF), 65504.0f32);
        assert_eq!(f16_to_f32(0xFBFF), -65504.0f32);
        assert!(f16_to_f32(0x7C00).is_infinite() && f16_to_f32(0x7C00) > 0.0);
        assert!(f16_to_f32(0xFC00).is_infinite() && f16_to_f32(0xFC00) < 0.0);
        let nan = f16_to_f32(0x7E00);
        assert!(nan.is_nan());
        assert_eq!(nan.to_bits() >> 31, 0);
    }

    #[test]
    fn dequantizes_q8_0() {
        let mut block = Vec::with_capacity(34);
        block.extend_from_slice(&0x4000u16.to_le_bytes()); // d = 2.0
        let qs: Vec<i8> = (-16..16).collect();
        block.extend(qs.iter().map(|&value| value as u8));
        let expected: Vec<f32> = qs.iter().map(|&value| 2.0 * f32::from(value)).collect();
        assert_eq!(dequantize_q8_0(&block), expected);
    }

    #[test]
    fn dequantizes_q4_k_both_scale_packing_branches() {
        let mut block = Vec::with_capacity(144);
        block.extend_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
        block.extend_from_slice(&0x3C00u16.to_le_bytes()); // dmin = 1.0
        let scales: [u8; 12] = [1, 2, 0, 0, 0, 3, 0, 0, 0x25, 0x37, 0, 0];
        block.extend_from_slice(&scales);
        block.extend(std::iter::repeat_n(0x21u8, 32));
        block.extend(std::iter::repeat_n(0x00u8, 32));
        block.extend(std::iter::repeat_n(0x21u8, 32));
        block.extend(std::iter::repeat_n(0x00u8, 32));

        let mut expected = Vec::with_capacity(256);
        expected.extend(std::iter::repeat_n(1.0f32 * 1.0 - 0.0, 32));
        expected.extend(std::iter::repeat_n(2.0f32 * 2.0 - 3.0, 32));
        expected.extend(std::iter::repeat_n(0.0f32, 64));
        expected.extend(std::iter::repeat_n(5.0f32 * 1.0 - 2.0, 32));
        expected.extend(std::iter::repeat_n(7.0f32 * 2.0 - 3.0, 32));
        expected.extend(std::iter::repeat_n(0.0f32, 64));
        assert_eq!(dequantize_q4_k(&block), expected);
    }

    #[test]
    fn dequantizes_q5_k_high_bit_plane() {
        let mut block = Vec::with_capacity(176);
        block.extend_from_slice(&0x3C00u16.to_le_bytes());
        block.extend_from_slice(&0x3C00u16.to_le_bytes());
        let scales: [u8; 12] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        block.extend_from_slice(&scales);
        let mut qh = [0u8; 32];
        qh[0] = 0x01;
        block.extend_from_slice(&qh);
        let mut qs = [0u8; 128];
        qs[0] = 0x01;
        block.extend_from_slice(&qs);

        let mut expected = vec![0.0f32; 256];
        expected[0] = 17.0;
        assert_eq!(dequantize_q5_k(&block), expected);
    }
}
