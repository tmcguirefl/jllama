# io/

| File | Milestone | Status |
|------|-----------|--------|
| `gguf.ijs` | M4+M13 | **done** - F16/F32 GGUF reader + Llama MHA/GQA model pack |
| `vocab.ijs` | M5+M10 | **done** - GPT-2 BPE + Llama SPM encode/decode |

## jllamagguf (M4)

| Verb | Role |
|------|------|
| `gguf_load` | path -> load box (meta, tensor infos, align, data_off, bytes) |
| `gguf_meta` | load meta key |
| `gguf_tensor` | load tensor name -> J f64 (jllama layout) |
| `gguf_names` | list tensor names |
| `model_from_gguf` | path -> jllama model box |
| `gguf_summary` | dump header/tensors |

**Supported:** little-endian GGUF v3, `GGML_TYPE_F32` / `F16` only, Llama-arch dense MHA or GQA (`n_head_kv` divides `n_head`).

**Layout:** ggml 2d `ne0` contiguous; non-embd weights become `n_in x n_out` for `x +/ . * w`. `token_embd.weight` stays `n_vocab x n_embd`.

**Fixture:** `test/fixtures/tiny_llama_f16.gguf` (regen via `python3 labs/make_fixture_gguf.py`).

```j
NB. after: jconsole jllama_dev.ijs
loadcore_jllama_ ''
m =. model_from_gguf_jllamagguf_ jllama_root '' , 'test/fixtures/tiny_llama_f16.gguf'
m generate_jllamamodel_ (0 1) ; 3
```

## jllamavocab (M5)

| Verb | Role |
|------|------|
| `vocab_from_gguf` | path -> vocab box from `tokenizer.ggml.*` |
| `vocab_from_load` | gguf load box -> vocab box |
| `encode` | `vocab encode text` -> int ids |
| `decode` | `vocab decode ids` -> text |
| `vocab_token` / `vocab_bos` / `vocab_eos` / `vocab_unk` | helpers |

**Supported:**
- `tokenizer.ggml.model` = `gpt2` or `bpe` — whole-string byte BPE (GPT-2 `bytes_to_unicode`); `pre=llama-bpe` defaults `add_bos=1` when meta omits it (Llama 3)
- `tokenizer.ggml.model` = `llama` — SentencePiece-style SPM (llama.cpp `llm_tokenizer_spm`), scores + U+2581 space escape, byte fallback `<0xNN>`

No full GPT-2 regex pre-tokenizer yet.

**Fixture:** `test/fixtures/tiny_bpe_vocab.gguf` (regen via `python3 labs/make_fixture_vocab.py`).

**Parity fixture (M6):** `test/fixtures/tiny_parity_f16.gguf` combines weights + tokenizer (regen via `python3 labs/make_fixture_parity.py`).

```j
loadcore_jllama_ ''
v =. vocab_from_gguf_jllamavocab_ jllama_root '' , 'test/fixtures/tiny_bpe_vocab.gguf'
ids =. v encode_jllamavocab_ 'ab'
v decode_jllamavocab_ ids
```
