# gguf-model-ingestion Specification

## Purpose
TBD - created by archiving change wire-gguf-into-model-loading. Update Purpose after archive.
## Requirements
### Requirement: GGUF Ingestion Is Scoped To A Real Model Component's Architecture

The GGUF ingestor SHALL accept only GGUF files declaring `general.architecture` values a real Magnetar Model Component exists for (`qwen2`), and SHALL reject any other declared architecture with a structured error naming it, rather than guessing at an unsupported graph shape.

#### Scenario: Supported architecture ingests

- **WHEN** a GGUF file declares `general.architecture: "qwen2"`
- **THEN** ingestion proceeds to normalize architecture configuration and tensors

#### Scenario: Unsupported architecture is rejected

- **WHEN** a GGUF file declares an architecture other than `qwen2`
- **THEN** ingestion returns a structured error naming the declared architecture, and no tensor materialization is attempted

---

### Requirement: GGUF Tensor Shape And Layout Match The Runtime's Existing Convention

The GGUF ingestor SHALL normalize each tensor's shape and, where applicable, byte layout to exactly the same convention the existing Hugging Face/Safetensors ingestion path produces for the equivalent tensor, so the Runtime and Model Component require no format-specific handling.

#### Scenario: A projection weight's logical shape matches the Hugging Face convention

- **WHEN** a GGUF file's 2D projection tensor (`q`/`k`/`v`/output attention projections, MLP gate/up/down projections, or the output/`lm_head` projection) is normalized
- **THEN** its resulting shape and byte layout are `[in_features, out_features]`, identical in convention to what Hugging Face/Safetensors ingestion of the equivalent checkpoint would produce

#### Scenario: The embedding table is never transposed

- **WHEN** a GGUF file's token embedding tensor is normalized
- **THEN** its resulting shape is `[vocab_size, hidden_size]`, matching Hugging Face/Safetensors ingestion's identical treatment of the same tensor as a lookup table, not a projection

#### Scenario: Identical real weight values ingested through either format produce identical generation

- **WHEN** the same real (non-trivial) weight values are ingested once through GGUF ingestion and once through Hugging Face/Safetensors ingestion, and generation is run against identical input token ids on both
- **THEN** the generated token ids are identical

---

### Requirement: A Tied-Embedding GGUF Checkpoint's Output Projection Is Usable Either Way

Whether a GGUF file declares an explicit output/`lm_head` projection tensor or omits it for a tied-embedding checkpoint, the GGUF ingestor SHALL produce a usable `lm_head` tensor entry: the declared one if present, or one derived from the token embedding tensor if absent.

#### Scenario: An explicit output projection is used directly

- **WHEN** a GGUF file declares its own output projection tensor
- **THEN** ingestion uses that tensor's own real bytes for `lm_head`, without deriving a synthetic one

#### Scenario: A missing output projection is derived from the token embedding

- **WHEN** a GGUF file declares no output projection tensor
- **THEN** ingestion derives a `lm_head` tensor from the token embedding tensor's own real bytes, transposed to the same logical shape a declared projection would have

---

### Requirement: GGUF Ingestion Builds A Real Tokenizer From The File's Own Embedded Vocabulary

The GGUF ingestor's tokenizer construction SHALL build a real tokenizer directly from the GGUF file's own embedded vocabulary and merge rules, without requiring a separate tokenizer file, and SHALL satisfy the same special-token uniqueness constraint every tokenizer in the Runtime satisfies regardless of source format.

#### Scenario: A real tokenizer builds from embedded vocabulary alone

- **WHEN** a GGUF file declares a byte-level BPE tokenizer model with embedded token and merge arrays
- **THEN** a real tokenizer builds from those arrays alone, without reading any file other than the GGUF file itself

#### Scenario: A shared beginning-of-sequence and end-of-sequence id does not fail construction

- **WHEN** a GGUF file declares the same token id for both its beginning-of-sequence and end-of-sequence special tokens
- **THEN** tokenizer construction succeeds, registering that id under one role rather than failing on a special-token conflict

#### Scenario: A byte-level BPE adds no space before the text

- **WHEN** a byte-level BPE tokenizer built from a GGUF file encodes text that does not start with a space
- **THEN** the first token is the text's own first piece, not one carrying a leading space, as llama.cpp and the checkpoint's upstream `tokenizer.json` encode it

### Requirement: GGUF Ingestion Dequantizes Supported Block Formats

The GGUF ingestor SHALL dequantize a tensor whose declared storage uses a supported GGUF block quantization format (`Q8_0`, `Q4_K`, or `Q5_K`) to `F32` before that tensor's shape/layout normalization (including any projection-weight transpose) runs, and SHALL report that tensor's dequantized `F32` storage dtype, byte size, and absence of quantization metadata in the returned Model Artifact -- never the original quantized values. Any GGUF `ggml_type` this ingestor's underlying parser does not itself recognize continues to fail at the parser layer, before this ingestor ever sees such a tensor.

#### Scenario: A supported quantized tensor dequantizes before materialization

- **WHEN** a GGUF file declares a tensor with a supported quantized `ggml_type` (`Q4_K`, `Q5_K`, or `Q8_0`)
- **THEN** the returned tensor's storage dtype is `F32`, its byte size matches its dequantized element count, it carries no quantization metadata, and reading its payload returns the real dequantized values

#### Scenario: A quantized projection weight transposes correctly

- **WHEN** a supported quantized tensor is also a 2D projection weight subject to this ingestor's existing projection-weight transpose
- **THEN** dequantization completes before the transpose runs, and the transposed result matches what transposing the equivalent already-`F32` values would produce

#### Scenario: Unquantized tensors of a mixed file still ingest

- **WHEN** a GGUF file's tensors are entirely unquantized (`F32`/`F16`/`BF16`)
- **THEN** ingestion proceeds normally, unaffected by dequantization support existing for other tensors

### Requirement: GGUF Ingestion Derives A Layered File's Packed Tensors

GGUF ingestion SHALL derive the tensors a `gemma4` file's Model Component requests from its packed tensors:
- **Per-layer input tables:** each layer's slice of `per_layer_token_embd` (columns) and `per_layer_model_proj` (rows).
- **Routed experts:** the stacked gate and up projections from `ffn_gate_up_exps`.
- **Router:** `ffn_gate_inp` with `ffn_gate_inp.scale` and `hidden^-0.5` folded in, as `f32`, and `ffn_down_exps.scale` as the per-expert scale.

A block-quantized table SHALL be cut on block boundaries and copied without re-encoding, and a cut inside a block SHALL be refused.

#### Scenario: Block-quantized per-layer embedding
- **WHEN** `per_layer_token_embd` is `Q8_0` with one 32-element block per layer per row
- **THEN** `layers.1.per_layer_embedding` is each row's second block, byte for byte

### Requirement: GGUF Ingestion Reads Q4_0 Tensors

GGUF ingestion SHALL read a `Q4_0` tensor (GGML type 2) as storage dtype `Q4`.

A `Q4_0` projection SHALL keep its stored bytes, packed, and SHALL declare transposed storage, as a `Q8_0` projection does.

#### Scenario: A Q4_0 projection
- **WHEN** a GGUF file stores a projection weight as `Q4_0`
- **THEN** its tensor keeps storage dtype `Q4` with transposed storage declared, and Model Loading reads its values as `d * (q - 8)`, transposed

### Requirement: GGUF Ingestion Reads deepseek2 Files

GGUF ingestion SHALL read a `deepseek2` file's `attention.kv_lora_rank` as multi-head latent attention, its YaRN keys as llama.cpp applies them, its leading dense layers and routed experts with their routing, and SHALL split its latent-attention projections into the tensors the Model Component requests, from either form of the key and value projection llama.cpp's converter has written.

The expanded head widths come from `key_length_mla` and `value_length_mla` when present (files converted after llama.cpp split `attn_kv_b`, which declare the latent's widths in `key_length` and `value_length` and one K/V head), from `key_length` and `value_length` otherwise; the rotated width from `rope.dimension_count`. YaRN keeps cosines and sines unscaled and takes `mscale_all_dim` as `yarn_log_multiplier / 0.1`, with `beta_fast` 32 and `beta_slow` 1 unless declared. Scoring is softmax when `expert_gating_func` is absent or `1`, sigmoid when `2`; another value is refused. With `expert_group_count` above 1, a group is scored by its best expert after a softmax and by its two best after a sigmoid. `exp_probs_b.bias` is the selection bias. `attn_q`/`attn_q_b`, `attn_kv_a_mqa` and `attn_kv_b` are split by rows like the Hugging Face checkpoint's; `attn_v_b` is already the value projection; `attn_k_b`'s heads are transposed back, in the stored float type or in `f32` from a block-quantized one.

#### Scenario: A DeepSeek-V2-Lite file from before the split
- **WHEN** the loader reads the metadata of `DeepSeek-V2-Lite-Chat.Q4_K_M.gguf` (converted in 2024)
- **THEN** it declares the same latent attention, head widths, experts and dense first layer as the Hugging Face configuration, YaRN with factor 40 over 4096 positions and `mscale_all_dim` 0.707, and every tensor name in the file has a canonical name

#### Scenario: Either key and value form
- **WHEN** a file stores `attn_kv_b`, or `attn_k_b` and `attn_v_b` written from the same `kv_b` as the converter writes them
- **THEN** both yield the same split key and value projections, equal to the rows of `kv_b`

#### Scenario: A block-quantized split key projection
- **WHEN** `attn_k_b` is stored as `Q8_0`
- **THEN** the key projection reaches the Component in `f32`, each head transposed back

### Requirement: GGUF Loader Reads The Gemma 4 Tokenizer

The GGUF loader SHALL build the tokenizer of a file whose `tokenizer.ggml.model` is `gemma4` from its own metadata: its pieces and `tokenizer.ggml.merges` as a byte-fallback BPE, spaces encoded as `▁` with no space added before the text, `▁` decoded back to spaces. Its encoding and decoding SHALL match the checkpoint's upstream `tokenizer.json`.

#### Scenario: Gemma 4 E2B tokenizer parity
- **WHEN** the tokenizer is built from the Gemma 4 E2B QAT `Q4_0` file's metadata
- **THEN** it encodes and decodes the parity texts exactly as the upstream `tokenizer.json` does, allowing only the beginning-of-sequence token it adds before an encoding with special tokens
