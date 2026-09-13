//! Real, GGUF-embedded-vocabulary [`Tokenizer`] implementation. A GGUF
//! file embeds its tokenizer's full vocabulary/merges directly in its
//! key-value metadata (`tokenizer.ggml.tokens`/`tokenizer.ggml.merges`,
//! for a byte-level BPE model, `tokenizer.ggml.model == "gpt2"` -- Qwen2's
//! real tokenizer shape) rather than shipping a separate `tokenizer.json`
//! file, so this constructs a real `tokenizers::Tokenizer` programmatically
//! via `BPE::builder().vocab_and_merges(...)` instead of parsing bytes.
//! Encode/decode delegate to that real tokenizer exactly like
//! `loaders/huggingface::HuggingFaceTokenizer` does; the only difference
//! is construction. Kept entirely outside `magnetar-runtime` -- this crate
//! and `loaders/huggingface` each depend on `tokenizers` independently.

use magnetar_format_gguf::GgufMetadataValue;
use magnetar_runtime::ModelDigest;
use magnetar_runtime::production_model_ingestion::ProductionIngestionError;
use magnetar_runtime::tokenizer::{
    DecodeInput, DecodeOutput, EncodeInput, EncodeOutput, SpecialToken, SpecialTokenKind,
    TokenIdRange, TokenOffset, Tokenizer, TokenizerArtifactId, TokenizerError, TokenizerFamily,
    TokenizerId, TokenizerMetadata, TokenizerRevision, TruncationPolicy,
};
use std::collections::BTreeMap;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer as HfTokenizerImpl};

fn malformed(reason: impl Into<String>) -> ProductionIngestionError {
    ProductionIngestionError::MalformedMetadata {
        reason: reason.into(),
    }
}

fn string_array<'a>(
    metadata: &'a BTreeMap<String, GgufMetadataValue>,
    key: &str,
) -> Option<Vec<&'a str>> {
    match metadata.get(key) {
        Some(GgufMetadataValue::Array(values)) => Some(
            values
                .iter()
                .filter_map(|value| match value {
                    GgufMetadataValue::String(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect(),
        ),
        _ => None,
    }
}

fn uint_value(metadata: &BTreeMap<String, GgufMetadataValue>, key: &str) -> Option<u64> {
    match metadata.get(key) {
        Some(GgufMetadataValue::UInt32(value)) => Some(u64::from(*value)),
        Some(GgufMetadataValue::UInt64(value)) => Some(*value),
        _ => None,
    }
}

/// Real, GGUF-embedded-vocabulary [`Tokenizer`] implementation
/// (Tokenizer Contract). Encode/decode delegate to a real
/// `tokenizers::Tokenizer` built from this GGUF file's own vocabulary.
pub struct GgufTokenizer {
    inner: HfTokenizerImpl,
    metadata: TokenizerMetadata,
}

impl GgufTokenizer {
    /// Builds a real tokenizer from a GGUF file's own key-value metadata.
    /// Requires `tokenizer.ggml.model == "gpt2"` (byte-level BPE, Qwen2's
    /// real tokenizer shape) and non-empty `tokenizer.ggml.tokens`/
    /// `tokenizer.ggml.merges` arrays. Validates the vocabulary against
    /// `expected_vocab_size` (the embedding table's real row count)
    /// exactly like `HuggingFaceTokenizer::from_bytes` does, for the same
    /// reason: a tokenizer vocabulary larger than the embedding table
    /// would let it produce a token id indexing past real weight rows.
    pub fn from_gguf_metadata(
        metadata: &BTreeMap<String, GgufMetadataValue>,
        artifact_id: impl Into<String>,
        expected_vocab_size: Option<u64>,
    ) -> Result<Self, ProductionIngestionError> {
        let model_kind = match metadata.get("tokenizer.ggml.model") {
            Some(GgufMetadataValue::String(value)) => value.as_str(),
            _ => {
                return Err(malformed(
                    "GGUF metadata is missing required string key 'tokenizer.ggml.model'",
                ));
            }
        };
        if model_kind != "gpt2" {
            return Err(ProductionIngestionError::UnsupportedFormat {
                reason: format!(
                    "GGUF tokenizer.ggml.model is '{model_kind}'; only 'gpt2' (byte-level BPE, \
                     Qwen2's real tokenizer shape) is supported"
                ),
            });
        }

        let tokens = string_array(metadata, "tokenizer.ggml.tokens")
            .ok_or_else(|| malformed("GGUF metadata is missing required key 'tokenizer.ggml.tokens'"))?;
        if tokens.is_empty() {
            return Err(malformed("GGUF metadata's tokenizer.ggml.tokens is empty"));
        }
        let merges_raw = string_array(metadata, "tokenizer.ggml.merges")
            .ok_or_else(|| malformed("GGUF metadata is missing required key 'tokenizer.ggml.merges'"))?;

        let vocabulary_size = tokens.len() as u32;
        if let Some(expected) = expected_vocab_size
            && u64::from(vocabulary_size) > expected
        {
            return Err(malformed(format!(
                "GGUF tokenizer.ggml.tokens vocabulary size {vocabulary_size} exceeds the \
                 model's real embedding table row count {expected}"
            )));
        }

        let vocab: tokenizers::models::bpe::Vocab = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| ((*token).to_string(), id as u32))
            .collect();
        let merges: tokenizers::models::bpe::Merges = merges_raw
            .iter()
            .map(|entry| {
                entry
                    .split_once(' ')
                    .map(|(left, right)| (left.to_string(), right.to_string()))
                    .ok_or_else(|| {
                        malformed(format!(
                            "GGUF tokenizer.ggml.merges entry '{entry}' is not a space-separated pair"
                        ))
                    })
            })
            .collect::<Result<_, _>>()?;

        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .byte_fallback(false)
            .build()
            .map_err(|error| malformed(format!("GGUF vocabulary failed to build a BPE model: {error}")))?;

        let mut inner = HfTokenizerImpl::new(bpe);
        inner.with_pre_tokenizer(Some(ByteLevel::default()));
        inner.with_decoder(Some(ByteLevel::default()));

        // Every CONTROL-type token (llama.cpp's `token_type` enum value 3
        // -- BOS/EOS/PAD and similar) must be added as a special token so
        // it is never split by BPE merging and is correctly excluded by
        // `skip_special_tokens` at decode time, exactly like a real
        // `tokenizer.json`'s own `added_tokens` entries.
        if let Some(GgufMetadataValue::Array(token_types)) = metadata.get("tokenizer.ggml.token_type") {
            const CONTROL_TOKEN_TYPE: i32 = 3;
            let special_tokens: Vec<AddedToken> = token_types
                .iter()
                .enumerate()
                .filter_map(|(id, kind)| match kind {
                    GgufMetadataValue::Int32(value) if *value == CONTROL_TOKEN_TYPE => {
                        tokens.get(id).map(|text| AddedToken::from(*text, true))
                    }
                    _ => None,
                })
                .collect();
            if !special_tokens.is_empty() {
                inner.add_special_tokens(&special_tokens);
            }
        }

        let mut special_tokens = Vec::new();
        let mut push_special = |kind: SpecialTokenKind, id: Option<u64>| {
            let Some(id) = id.and_then(|id| u32::try_from(id).ok()) else {
                return;
            };
            let Some(text) = tokens.get(id as usize) else {
                return;
            };
            special_tokens.push(SpecialToken::new(kind, *text, id));
        };
        push_special(
            SpecialTokenKind::Bos,
            uint_value(metadata, "tokenizer.ggml.bos_token_id"),
        );
        push_special(
            SpecialTokenKind::Eos,
            uint_value(metadata, "tokenizer.ggml.eos_token_id"),
        );
        push_special(
            SpecialTokenKind::Pad,
            uint_value(metadata, "tokenizer.ggml.padding_token_id"),
        );

        let max_id = special_tokens
            .iter()
            .map(|token| token.id)
            .max()
            .unwrap_or(0)
            .max(vocabulary_size.saturating_sub(1));

        let artifact_id = artifact_id.into();
        let tokenizer_metadata = TokenizerMetadata {
            id: TokenizerId::new(&artifact_id)
                .map_err(|error| malformed(format!("invalid tokenizer id '{artifact_id}': {error}")))?,
            artifact: TokenizerArtifactId::new(&artifact_id)
                .map_err(|error| malformed(format!("invalid tokenizer artifact id '{artifact_id}': {error}")))?,
            digest: ModelDigest::sha256(
                tokens
                    .iter()
                    .flat_map(|token| token.as_bytes())
                    .copied()
                    .collect::<Vec<u8>>()
                    .as_slice(),
            ),
            family: TokenizerFamily::new("gguf-gpt2-bpe").map_err(|error| malformed(error.to_string()))?,
            revision: TokenizerRevision::new("1").map_err(|error| malformed(error.to_string()))?,
            vocabulary_size,
            added_token_count: inner.get_added_tokens_decoder().len() as u32,
            token_id_range: TokenIdRange::new(0, max_id),
            model_max_length: None,
            special_tokens,
            additional_special_tokens: Vec::new(),
            byte_fallback: false,
            normalization: None,
            pre_tokenizer: None,
            supports_offsets: true,
            supports_token_type_ids: false,
            supports_browser: false,
        };
        tokenizer_metadata
            .validate()
            .map_err(|error| malformed(format!("tokenizer metadata failed validation: {error}")))?;

        Ok(Self {
            inner,
            metadata: tokenizer_metadata,
        })
    }
}

impl Tokenizer for GgufTokenizer {
    fn metadata(&self) -> &TokenizerMetadata {
        &self.metadata
    }

    fn encode(&self, input: EncodeInput) -> Result<EncodeOutput, TokenizerError> {
        if input.return_offsets && !self.metadata.supports_offsets {
            return Err(TokenizerError::OffsetsUnsupported);
        }
        let encoding = self
            .inner
            .encode(input.text.as_str(), input.add_special_tokens)
            .map_err(|error| TokenizerError::BatchInputInvalid {
                message: error.to_string(),
            })?;
        let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let mut offsets: Option<Vec<TokenOffset>> = input.return_offsets.then(|| {
            encoding
                .get_offsets()
                .iter()
                .map(|(start, end)| TokenOffset {
                    byte_start: *start as u32,
                    byte_end: *end as u32,
                    char_start: Some(*start as u32),
                    char_end: Some(*end as u32),
                })
                .collect()
        });

        let limit = input.max_tokens;
        if let Some(limit) = limit
            && token_ids.len() > limit
        {
            match input.truncation {
                TruncationPolicy::None => {
                    return Err(TokenizerError::PromptTooLong {
                        token_count: token_ids.len(),
                        limit,
                    });
                }
                TruncationPolicy::Left | TruncationPolicy::ClientPolicy => {
                    let drop = token_ids.len() - limit;
                    token_ids.drain(0..drop);
                    if let Some(offsets) = &mut offsets {
                        offsets.drain(0..drop);
                    }
                }
                TruncationPolicy::Right | TruncationPolicy::ModelDefault => {
                    token_ids.truncate(limit);
                    if let Some(offsets) = &mut offsets {
                        offsets.truncate(limit);
                    }
                }
                TruncationPolicy::Middle => {
                    let left = limit / 2;
                    let right = limit - left;
                    let tail_start = token_ids.len() - right;
                    let mut truncated = token_ids[..left].to_vec();
                    truncated.extend_from_slice(&token_ids[tail_start..]);
                    token_ids = truncated;
                    if let Some(offsets) = &mut offsets {
                        let mut truncated = offsets[..left].to_vec();
                        truncated.extend_from_slice(&offsets[tail_start..]);
                        *offsets = truncated;
                    }
                }
            }
        }

        Ok(EncodeOutput {
            token_count: token_ids.len(),
            attention_mask: Some(vec![1u8; token_ids.len()]),
            token_type_ids: None,
            token_ids,
            offsets,
            diagnostics: Vec::new(),
        })
    }

    fn decode(&self, input: DecodeInput) -> Result<DecodeOutput, TokenizerError> {
        for id in &input.token_ids {
            if !self.metadata.token_id_range.contains(*id) {
                return Err(TokenizerError::InvalidTokenId { token_id: *id });
            }
        }
        let text = self
            .inner
            .decode(&input.token_ids, input.skip_special_tokens)
            .map_err(|error| TokenizerError::InvalidUtf8 {
                message: error.to_string(),
            })?;
        Ok(DecodeOutput {
            text,
            consumed_token_count: input.token_ids.len(),
            pending_partial_state: None,
            diagnostics: Vec::new(),
        })
    }
}
