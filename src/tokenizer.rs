//! Real, GGUF-embedded-vocabulary [`Tokenizer`] implementation. A GGUF
//! file embeds its tokenizer's full vocabulary directly in its key-value
//! metadata -- `tokenizer.ggml.tokens` with `tokenizer.ggml.merges` for a
//! byte-level BPE (`tokenizer.ggml.model == "gpt2"`, Qwen2's real
//! tokenizer shape), or with `tokenizer.ggml.scores` for a SentencePiece
//! BPE (`"llama"`, Llama 2 and Mistral) -- rather than shipping a separate
//! `tokenizer.json` file, so this constructs a real `tokenizers::Tokenizer`
//! programmatically via `BPE::builder().vocab_and_merges(...)` instead of
//! parsing bytes.
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
use tokenizers::decoders::byte_fallback::ByteFallback;
use tokenizers::decoders::fuse::Fuse;
use tokenizers::decoders::sequence::Sequence as DecoderSequence;
use tokenizers::decoders::strip::Strip;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::replace::Replace;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::processors::template::TemplateProcessing;
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

/// A byte-level BPE (`tokenizer.ggml.model == "gpt2"`, Qwen2's real
/// tokenizer shape) from the vocabulary and `tokenizer.ggml.merges`.
fn byte_level_bpe(
    metadata: &BTreeMap<String, GgufMetadataValue>,
    vocab: tokenizers::models::bpe::Vocab,
) -> Result<HfTokenizerImpl, ProductionIngestionError> {
    let merges_raw = string_array(metadata, "tokenizer.ggml.merges").ok_or_else(|| {
        malformed("GGUF metadata is missing required key 'tokenizer.ggml.merges'")
    })?;
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
        .map_err(|error| {
            malformed(format!(
                "GGUF vocabulary failed to build a BPE model: {error}"
            ))
        })?;
    let mut inner = HfTokenizerImpl::new(bpe);
    inner.with_pre_tokenizer(Some(ByteLevel::default()));
    inner.with_decoder(Some(ByteLevel::default()));
    Ok(inner)
}

/// llama.cpp's `token_type` for an ordinary vocabulary piece.
const NORMAL_TOKEN_TYPE: i32 = 1;

/// A SentencePiece BPE (`tokenizer.ggml.model == "llama"`): GGUF stores
/// its pieces with their scores and no merges. SentencePiece BPE merges,
/// at each step, the adjacent pair whose concatenation is the
/// highest-scoring piece; the equivalent ranked merge list pairs every
/// normal piece's splits into two pieces, ranked by the merged piece's
/// score (as `tokenizers`' own SentencePiece conversion does). Pieces
/// missing from the vocabulary fall back to their `<0xXX>` byte pieces;
/// `▁` stands for a space and starts the text, as the upstream
/// `tokenizer.json` does (`Metaspace`, prepended to the first segment).
fn sentencepiece_bpe(
    metadata: &BTreeMap<String, GgufMetadataValue>,
    tokens: &[&str],
    vocab: tokenizers::models::bpe::Vocab,
) -> Result<HfTokenizerImpl, ProductionIngestionError> {
    let scores: Vec<f32> = match metadata.get("tokenizer.ggml.scores") {
        Some(GgufMetadataValue::Array(values)) => values
            .iter()
            .map(|value| match value {
                GgufMetadataValue::Float32(score) => Ok(*score),
                _ => Err(malformed(
                    "GGUF tokenizer.ggml.scores holds a non-f32 value",
                )),
            })
            .collect::<Result<_, _>>()?,
        _ => {
            return Err(malformed(
                "GGUF metadata is missing required key 'tokenizer.ggml.scores'",
            ));
        }
    };
    let token_types: Vec<i32> = match metadata.get("tokenizer.ggml.token_type") {
        Some(GgufMetadataValue::Array(values)) => values
            .iter()
            .map(|value| match value {
                GgufMetadataValue::Int32(kind) => *kind,
                _ => NORMAL_TOKEN_TYPE,
            })
            .collect(),
        _ => vec![NORMAL_TOKEN_TYPE; tokens.len()],
    };
    if scores.len() != tokens.len() || token_types.len() != tokens.len() {
        return Err(malformed(format!(
            "GGUF declares {} tokens, {} scores and {} token types",
            tokens.len(),
            scores.len(),
            token_types.len()
        )));
    }

    // (merged score, merged id, left id, right id, left, right).
    let mut ranked = Vec::new();
    for (id, piece) in tokens.iter().enumerate() {
        if token_types[id] != NORMAL_TOKEN_TYPE {
            continue;
        }
        for (split, _) in piece.char_indices().skip(1) {
            let (left, right) = piece.split_at(split);
            if let (Some(&left_id), Some(&right_id)) = (vocab.get(left), vocab.get(right)) {
                ranked.push((scores[id], id as u32, left_id, right_id, left, right));
            }
        }
    }
    ranked.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
            .then(a.3.cmp(&b.3))
    });
    let merges: tokenizers::models::bpe::Merges = ranked
        .into_iter()
        .map(|(_, _, _, _, left, right)| (left.to_string(), right.to_string()))
        .collect();

    let mut builder = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .byte_fallback(true)
        .fuse_unk(true);
    if let Some(unknown) = uint_value(metadata, "tokenizer.ggml.unknown_token_id")
        .and_then(|id| tokens.get(id as usize))
    {
        builder = builder.unk_token((*unknown).to_string());
    }
    let bpe = builder.build().map_err(|error| {
        malformed(format!(
            "GGUF vocabulary failed to build a SentencePiece BPE model: {error}"
        ))
    })?;
    let mut inner = HfTokenizerImpl::new(bpe);
    inner.with_pre_tokenizer(Some(Metaspace::new(
        '\u{2581}',
        PrependScheme::First,
        false,
    )));
    let space = Replace::new("\u{2581}", " ")
        .map_err(|error| malformed(format!("the SentencePiece decoder failed: {error}")))?;
    inner.with_decoder(Some(DecoderSequence::new(vec![
        space.into(),
        ByteFallback::new().into(),
        Fuse::new().into(),
        Strip::new(' ', 1, 0).into(),
    ])));
    Ok(inner)
}

/// Real, GGUF-embedded-vocabulary [`Tokenizer`] implementation
/// (Tokenizer Contract). Encode/decode delegate to a real
/// `tokenizers::Tokenizer` built from this GGUF file's own vocabulary.
pub struct GgufTokenizer {
    inner: HfTokenizerImpl,
    metadata: TokenizerMetadata,
}

impl GgufTokenizer {
    /// Builds a real tokenizer from a GGUF file's own key-value metadata:
    /// `tokenizer.ggml.model == "gpt2"` (byte-level BPE from
    /// `tokenizer.ggml.merges`, Qwen2's real tokenizer shape) or `"llama"`
    /// (SentencePiece BPE from the pieces' `tokenizer.ggml.scores`, with
    /// byte fallback). Validates the vocabulary against
    /// `expected_vocab_size` (the embedding table's real row count)
    /// exactly like `HuggingFaceTokenizer::from_bytes` does, for the same
    /// reason: a tokenizer vocabulary larger than the embedding table
    /// would let it produce a token id indexing past real weight rows.
    /// `tokenizer.ggml.add_bos_token` prefixes encodings that add special
    /// tokens with the beginning-of-sequence token.
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
        let tokens = string_array(metadata, "tokenizer.ggml.tokens").ok_or_else(|| {
            malformed("GGUF metadata is missing required key 'tokenizer.ggml.tokens'")
        })?;
        if tokens.is_empty() {
            return Err(malformed("GGUF metadata's tokenizer.ggml.tokens is empty"));
        }
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

        let (mut inner, family, byte_fallback) = match model_kind {
            "gpt2" => (byte_level_bpe(metadata, vocab)?, "gguf-gpt2-bpe", false),
            "llama" => (
                sentencepiece_bpe(metadata, &tokens, vocab)?,
                "gguf-llama-spm",
                true,
            ),
            other => {
                return Err(ProductionIngestionError::UnsupportedFormat {
                    reason: format!(
                        "GGUF tokenizer.ggml.model is '{other}'; only 'gpt2' (byte-level BPE) \
                         and 'llama' (SentencePiece BPE) are supported"
                    ),
                });
            }
        };

        // Every CONTROL-type token (llama.cpp's `token_type` enum value 3
        // -- BOS/EOS/PAD and similar) must be added as a special token so
        // it is never split by BPE merging and is correctly excluded by
        // `skip_special_tokens` at decode time, exactly like a real
        // `tokenizer.json`'s own `added_tokens` entries.
        if let Some(GgufMetadataValue::Array(token_types)) =
            metadata.get("tokenizer.ggml.token_type")
        {
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
                inner.add_special_tokens(special_tokens).map_err(|error| {
                    malformed(format!("GGUF control tokens failed to register: {error}"))
                })?;
            }
        }

        // `tokenizer.ggml.add_bos_token`: the checkpoint's prompts start
        // with its beginning-of-sequence token, as the upstream
        // `tokenizer.json`'s template processor adds it.
        if matches!(
            metadata.get("tokenizer.ggml.add_bos_token"),
            Some(GgufMetadataValue::Bool(true))
        ) && let Some(bos) = uint_value(metadata, "tokenizer.ggml.bos_token_id")
            .and_then(|id| u32::try_from(id).ok())
            && let Some(text) = tokens.get(bos as usize)
        {
            let processor = TemplateProcessing::builder()
                .try_single(format!("{text} $A"))
                .and_then(|builder| {
                    builder
                        .special_tokens(vec![((*text).to_string(), bos)])
                        .build()
                        .map_err(|error| error.to_string())
                })
                .map_err(|error| {
                    malformed(format!(
                        "the beginning-of-sequence template failed: {error}"
                    ))
                })?;
            inner.with_post_processor(Some(processor));
        }

        // `TokenizerMetadata::validate` rejects two special tokens sharing
        // the same id, regardless of kind -- a real constraint this
        // ingestor must satisfy exactly like `HuggingFaceTokenizer` does,
        // not a GGUF-specific relaxation. A real Hugging Face
        // `tokenizer_config.json` for a Qwen2 checkpoint commonly *omits*
        // `bos_token` entirely (Qwen2 has no true distinct
        // beginning-of-sequence token), so `HuggingFaceTokenizer` never
        // even attempts to register one there -- but GGUF's key-value
        // schema always writes a `tokenizer.ggml.bos_token_id` (llama.cpp's
        // real writer convention defaults it to the same id as
        // `eos_token_id` when the source model has no distinct one,
        // discovered running this ingestor against a real downloaded
        // Qwen2.5-0.5B-Instruct GGUF file, whose bos/eos ids are
        // identical). Registering ids in eos, bos, pad order and skipping
        // one that would collide with an id already claimed reproduces
        // exactly what the Hugging Face path's own field-omission already
        // achieves for the same real model.
        let mut special_tokens = Vec::new();
        let mut claimed_ids = std::collections::BTreeSet::new();
        let mut push_special = |kind: SpecialTokenKind, id: Option<u64>| {
            let Some(id) = id.and_then(|id| u32::try_from(id).ok()) else {
                return;
            };
            if !claimed_ids.insert(id) {
                return;
            }
            let Some(text) = tokens.get(id as usize) else {
                return;
            };
            special_tokens.push(SpecialToken::new(kind, *text, id));
        };
        push_special(
            SpecialTokenKind::Eos,
            uint_value(metadata, "tokenizer.ggml.eos_token_id"),
        );
        push_special(
            SpecialTokenKind::Bos,
            uint_value(metadata, "tokenizer.ggml.bos_token_id"),
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
            id: TokenizerId::new(&artifact_id).map_err(|error| {
                malformed(format!("invalid tokenizer id '{artifact_id}': {error}"))
            })?,
            artifact: TokenizerArtifactId::new(&artifact_id).map_err(|error| {
                malformed(format!(
                    "invalid tokenizer artifact id '{artifact_id}': {error}"
                ))
            })?,
            digest: ModelDigest::sha256(
                tokens
                    .iter()
                    .flat_map(|token| token.as_bytes())
                    .copied()
                    .collect::<Vec<u8>>()
                    .as_slice(),
            ),
            family: TokenizerFamily::new(family).map_err(|error| malformed(error.to_string()))?,
            revision: TokenizerRevision::new("1").map_err(|error| malformed(error.to_string()))?,
            vocabulary_size,
            added_token_count: inner.get_added_tokens_decoder().len() as u32,
            token_id_range: TokenIdRange::new(0, max_id),
            model_max_length: None,
            special_tokens,
            additional_special_tokens: Vec::new(),
            byte_fallback,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_metadata_with(
        bos: Option<u32>,
        eos: Option<u32>,
    ) -> BTreeMap<String, GgufMetadataValue> {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "tokenizer.ggml.model".into(),
            GgufMetadataValue::String("gpt2".into()),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".into(),
            GgufMetadataValue::Array(vec![
                GgufMetadataValue::String("<|endoftext|>".into()),
                GgufMetadataValue::String("hello".into()),
                GgufMetadataValue::String("world".into()),
            ]),
        );
        metadata.insert(
            "tokenizer.ggml.merges".into(),
            GgufMetadataValue::Array(Vec::new()),
        );
        if let Some(bos) = bos {
            metadata.insert(
                "tokenizer.ggml.bos_token_id".into(),
                GgufMetadataValue::UInt32(bos),
            );
        }
        if let Some(eos) = eos {
            metadata.insert(
                "tokenizer.ggml.eos_token_id".into(),
                GgufMetadataValue::UInt32(eos),
            );
        }
        metadata
    }

    /// A real, discovered bug: llama.cpp's GGUF writer always declares
    /// `tokenizer.ggml.bos_token_id`, defaulting it to the same id as
    /// `eos_token_id` for a model with no distinct BOS (Qwen2.5-0.5B-
    /// Instruct's real GGUF export does exactly this) --
    /// `TokenizerMetadata::validate` rejects two special tokens sharing
    /// one id regardless of kind, so both must not be registered.
    #[test]
    fn shared_bos_eos_id_does_not_conflict() {
        let metadata = tiny_metadata_with(Some(0), Some(0));
        let tokenizer = GgufTokenizer::from_gguf_metadata(&metadata, "conflict-test", Some(3))
            .expect("a shared bos/eos id must not be treated as a validation conflict");
        assert!(
            tokenizer
                .metadata()
                .special_token(SpecialTokenKind::Eos)
                .is_some(),
            "eos must still be registered"
        );
        assert!(
            tokenizer
                .metadata()
                .special_token(SpecialTokenKind::Bos)
                .is_none(),
            "bos must be skipped once eos already claimed the same id"
        );
    }

    #[test]
    fn distinct_bos_and_eos_ids_are_both_registered() {
        let metadata = tiny_metadata_with(Some(0), Some(1));
        let tokenizer = GgufTokenizer::from_gguf_metadata(&metadata, "distinct-test", Some(3))
            .expect("distinct bos/eos ids build successfully");
        assert_eq!(
            tokenizer
                .metadata()
                .special_token(SpecialTokenKind::Bos)
                .map(|token| token.id),
            Some(0)
        );
        assert_eq!(
            tokenizer
                .metadata()
                .special_token(SpecialTokenKind::Eos)
                .map(|token| token.id),
            Some(1)
        );
    }
}

/// A real SentencePiece GGUF tokenizer against its upstream
/// `tokenizer.json`: set `MAGNETAR_GGUF_SPM_HEADER` to a prefix of a GGUF
/// file holding its whole header and `MAGNETAR_GGUF_SPM_TOKENIZER_JSON`
/// to the same checkpoint's `tokenizer.json`.
#[cfg(test)]
mod sentencepiece_parity {
    use super::*;

    #[test]
    #[ignore = "needs a real SentencePiece GGUF header and its tokenizer.json"]
    fn encodes_and_decodes_like_the_upstream_tokenizer_json() {
        let header = std::fs::read(std::env::var("MAGNETAR_GGUF_SPM_HEADER").unwrap()).unwrap();
        let artifact = magnetar_format_gguf::parse_prefix(&header, u64::MAX / 4).unwrap();
        let ours = GgufTokenizer::from_gguf_metadata(&artifact.metadata, "spm", None).unwrap();
        let upstream =
            HfTokenizerImpl::from_file(std::env::var("MAGNETAR_GGUF_SPM_TOKENIZER_JSON").unwrap())
                .unwrap();
        let corpus = [
            "The capital of France is",
            "Hello, world!",
            " leading space and  double  spaces",
            "[INST] What is 2+2? [/INST]",
            "naïve café — résumé 東京 🚀",
            "def f(x):\n    return x * 2\n",
            "Numbers 1234567 and 3.14159, e=mc^2.",
            "\t\ttabs\nnewlines\r\n",
            "Unbelievably, internationalization is antidisestablishmentarianism.",
            "",
        ];
        for text in corpus {
            for special in [true, false] {
                let expected = upstream.encode(text, special).unwrap().get_ids().to_vec();
                let mut input = EncodeInput::new(text);
                input.add_special_tokens = special;
                let actual = ours.encode(input).unwrap().token_ids;
                assert_eq!(
                    actual, expected,
                    "encoding {text:?} (special tokens {special})"
                );
                let decoded = ours
                    .decode(DecodeInput {
                        token_ids: actual.clone(),
                        skip_special_tokens: true,
                        clean_up_tokenization_spaces: false,
                        streaming_state: None,
                    })
                    .unwrap()
                    .text;
                assert_eq!(
                    decoded,
                    upstream.decode(&expected, true).unwrap(),
                    "decoding {text:?}"
                );
            }
        }
    }
}
