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
a real Magnetar Model Component exists for) and unquantized `F32`/`F16`/
`BF16` tensors only. A GGUF file declaring a quantized tensor (`Q4_K`/
`Q5_K`/`Q8_0`) is rejected structurally with a named error --
dequantization-at-load or quantized compute kernels are a separate,
not-yet-implemented Magnetar chantier.

## Status

**Real implementation**, not a fixture. `src/config.rs` normalizes GGUF's
`qwen2.*`/`tokenizer.ggml.*` key-value metadata into `magnetar-runtime`'s
generic `ModelArchitectureConfig`, with the same validation discipline
`loaders/huggingface::config` applies to `config.json` (missing/zero/
inconsistent values rejected structurally). `src/weights.rs` composes the
real `magnetar-format-gguf` parser, rejects quantized tensors, and
reverses GGUF's `ne[]` dimension order back to the Hugging Face-equivalent
row-major shape (GGML's `ne[]` is the reverse of PyTorch's shape labeling
over the *same* underlying bytes -- llama.cpp's real GGUF writer applies
no byte-level permutation for Qwen2, confirmed against its current
`conversion/qwen.py` source). `src/weight_layout.rs`/
`src/derived_lm_head.rs` are the same projection-weight transpose and
tied-`lm_head` derivation `loaders/huggingface` implements for the
equivalent Safetensors case (ported, not shared -- these are independent
externalized modules that cannot depend on each other). `src/tokenizer.rs`
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

**Verified two ways beyond unit tests:**
1. The exact same real (non-trivial) weight values, ingested through this
   crate and through `loaders/huggingface` respectively, produce
   byte-for-byte identical generation output -- the strongest available
   proof the shape-reversal/transpose logic is correct without a real
   downloaded checkpoint.
2. The real public `Qwen2.5-0.5B-Instruct-GGUF` checkpoint (an
   unquantized F16 export) loads and generates the exact same output as
   the already-verified `loaders/huggingface` path against the equivalent
   real Safetensors checkpoint, for the same prompt.

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
