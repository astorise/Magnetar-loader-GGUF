//! Block sizes of GGUF's block-quantized formats, used to validate a
//! tensor's element count at discovery. This crate no longer dequantizes:
//! a block-quantized tensor's original bytes reach Model Loading unchanged,
//! which either hands them to the Provider packed or dequantizes them with
//! `magnetar-runtime`'s own `Q8_0`/`Q4_K`/`Q5_K` implementation
//! (`execute-direct-quantized-weights`).

/// GGUF `block_q8_0`: 32 elements per block.
pub const Q8_0_BLOCK_ELEMENTS: u64 = 32;
/// GGUF K-quant super-block element count (`QK_K` upstream), shared by
/// `block_q4_K` and `block_q5_K`.
pub const QK_BLOCK_ELEMENTS: u64 = 256;
