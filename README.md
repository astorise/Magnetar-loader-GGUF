# magnetar-loader-gguf

## Purpose

An external, pinned production Model Artifact ingestor for the
[Magnetar](https://github.com/astorise/Magnetar) local AI Runtime
(`wire-gguf-into-model-loading`). Implements `magnetar-runtime`'s generic
`ProductionModelArtifactIngestor` contract for a single local GGUF file:
real architecture-config extraction from GGUF's own key-value metadata,
tensor discovery/renaming, and a real byte-level-BPE tokenizer built
programmatically from the GGUF file's own embedded vocabulary -- no
separate `config.json`/`tokenizer.json` files, unlike a Hugging Face
bundle (see `loaders/huggingface`).

Scoped to `general.architecture == "qwen2"` (the only architecture family
a real Magnetar Model Component exists for). A GGUF file may declare
`F32`/`F16`/`BF16` tensors, or `Q8_0`/`Q4_K`/`Q5_K` block-quantized
tensors -- the latter are dequantized to `F32` right here, before this
crate's own projection-weight transpose runs (`resolve-gguf-quantized-
projection-transpose-sequencing`), since that transpose assumes a flat
per-element byte width that block-quantized data does not have.
GPTQ/AWQ/BitsAndBytes (unrelated, Hugging Face/Safetensors-shaped
quantization schemes) remain out of scope entirely.

## Status

**Real implementation**, not a fixture. `src/config.rs` normalizes GGUF's
`qwen2.*`/`tokenizer.ggml.*` key-value metadata into `magnetar-runtime`'s
generic `ModelArchitectureConfig`, with the same validation discipline
`loaders/huggingface::config` applies to `config.json` (missing/zero/
inconsistent values rejected structurally). `src/weights.rs` composes the
real `magnetar-format-gguf` parser and reverses GGUF's `ne[]` dimension
order back to the Hugging Face-equivalent row-major shape (GGML's `ne[]`
is the reverse of PyTorch's shape labeling over the *same* underlying
bytes -- llama.cpp's real GGUF writer applies no byte-level permutation
for Qwen2, confirmed against its current `conversion/qwen.py` source); a
tensor quantized with `Q8_0`/`Q4_K`/`Q5_K` is dequantized to `F32` at
payload-read time via `src/dequantize.rs` (ported bit-for-bit from real
upstream `ggml-org/llama.cpp` `ggml-quants.c`, including the historically
error-prone K-quant sub-block scale/min packing `Q4_K`/`Q5_K` share), with
its declared `storage_dtype`/`size_bytes`/`quantization` overridden
accordingly so every later step treats it exactly like a real `F32`
tensor. `src/weight_layout.rs`/`src/derived_lm_head.rs` are the same
projection-weight transpose and tied-`lm_head` derivation
`loaders/huggingface` implements for the equivalent Safetensors case
(ported, not shared -- these are independent externalized modules that
cannot depend on each other) -- dequantizing *before* this transpose runs
is what makes a quantized projection weight transpose correctly instead
of having its block structure silently corrupted. `src/tokenizer.rs`
builds a real `tokenizers::Tokenizer` from the GGUF file's own
`tokenizer.ggml.tokens`/`merges` arrays via `BPE::builder().vocab_and_merges`
(kept out of `magnetar-runtime` entirely, matching `loaders/huggingface`'s
own use of the `tokenizers` crate).

Parsing/normalizing a bundle never grants trust: the `ModelManifest` this
crate produces still goes through `magnetar-runtime`'s own
`ModelManifest::validate` and `ModelTrustStore::evaluate` like any other
manifest before Model Loading may materialize anything from it. A declared
chat template (`tokenizer.chat_template`) is threaded into the manifest as
a named, digested part, exactly like `loaders/huggingface`'s own
convention -- the raw template text is not carried on the manifest; a
caller that wants to render it calls this crate's `load_gguf_chat_template`
and renders the returned text with whatever `ChatTemplateFormatter` it has
available (this crate does not depend on a Jinja2 engine itself).

**Verified beyond unit tests:**
1. The exact same real (non-trivial) weight values, ingested through this
   crate and through `loaders/huggingface` respectively, produce
   byte-for-byte identical generation output -- the strongest available
   proof the shape-reversal/transpose logic is correct without a real
   downloaded checkpoint.
2. The real public `Qwen2.5-0.5B-Instruct-GGUF` checkpoint (an
   unquantized F16 export) loads and generates the exact same output as
   the already-verified `loaders/huggingface` path against the equivalent
   real Safetensors checkpoint, for the same prompt.
3. A hand-constructed, non-square, genuinely `Q8_0`-quantized projection
   weight ingested end to end produces exactly the hand-computed
   dequantized-then-transposed values -- proving the dequantize-before-
   transpose sequencing is correct, not merely that it compiles or that a
   square (transpose-bug-masking) case happens to look right.

## Governing contract

[`production-model-ingestion`](https://github.com/astorise/Magnetar/blob/main/openspec/changes/implement-production-qwen-model-loading/specs/production-model-ingestion/spec.md)
in the main Magnetar repository's OpenSpec change set defines the generic
Runtime-owned contract this crate implements;
[`wire-gguf-into-model-loading`](https://github.com/astorise/Magnetar/blob/main/openspec/changes/archive/wire-gguf-into-model-loading/proposal.md)
defines this crate's own specific requirements.

## Relationship to magnetar-runtime

`magnetar-runtime` never imports this crate (enforced by CI's
`submodule-integration` dependency guard, covering every
`magnetar-loader-*` crate). An embedder registers a `GgufIngestor`
instance with `magnetar-runtime`'s generic `ProductionIngestionRegistry`;
Runtime performs every trust, memory, component, residency,
materialization, and readiness decision from there. It is pinned into the
main Magnetar repository as a git submodule at `loaders/gguf`.
