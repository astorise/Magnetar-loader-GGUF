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

/// The elements per block of a block-quantized storage dtype, `None` for
/// any other dtype.
pub fn block_elements(dtype: magnetar_runtime::model::ModelDType) -> Option<u64> {
    use magnetar_runtime::model::ModelDType;
    match dtype {
        ModelDType::Q8 | ModelDType::Iq4Nl => Some(Q8_0_BLOCK_ELEMENTS),
        ModelDType::Q2K
        | ModelDType::Q3K
        | ModelDType::Iq4Xs
        | ModelDType::Iq2Xxs
        | ModelDType::Iq2Xs
        | ModelDType::Iq2S
        | ModelDType::Iq3Xxs
        | ModelDType::Iq3S
        | ModelDType::Q4K
        | ModelDType::Q5K
        | ModelDType::Q6K => Some(QK_BLOCK_ELEMENTS),
        _ => None,
    }
}
