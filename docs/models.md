# Models

## Test fixture (M4)

| Field | Value |
|-------|--------|
| Path | `test/fixtures/tiny_llama_f16.gguf` |
| Arch | llama |
| Dtypes | F16 weights + F32 norms |
| Shape | n_vocab=8, n_embd=4, n_head=2, n_layer=1, n_ff=8 |
| Regen | `python3 labs/make_fixture_gguf.py` |
| Expect | `test/fixtures/tiny_llama_f16.expect.txt` |

Not a real LM — only for loader/shape/generate smoke.

## Parity fixture (M6)

| Field | Value |
|-------|--------|
| Path | `test/fixtures/tiny_parity_f16.gguf` |
| Arch | llama dense MHA |
| Dtypes | F16 weights + F32 norms |
| Shape | n_vocab=267, n_embd=64, n_head=4, n_layer=2, n_ff=128 |
| Tokenizer | GPT-2 byte BPE in same GGUF |
| Regen | `python3 labs/make_fixture_parity.py` |
| Meta | `test/fixtures/tiny_parity_f16.meta.txt` |
| Oracle | `tools/oracle_greedy` via `labs/run_oracle.sh` (libllama) |

## Primary lab model (M10)

| Field | Value |
|-------|--------|
| Path | `models/stories15M.F16.gguf` (gitignored) |
| Source | [shibatch/stories-converted](https://huggingface.co/shibatch/stories-converted) → `stories15M.F16.gguf` |
| Why this file | **Llama arch + MHA** (`n_head = n_head_kv = 6`), **F16**, ~47 MB — small English lab; GQA supported from M13 |
| Arch | llama dense MHA |
| Hparams | n_vocab=32000, n_embd=288, n_head=6, n_layer=6, n_ff=768, rope θ=10000, rms_eps=1e-5, ctx=128 |
| Tokenizer | `tokenizer.ggml.model = llama` (SentencePiece-style SPM) |
| Training domain | TinyStories-style English |
| Driver | `./jllama_cli.ijs` |
| Oracle | `tools/oracle_tokenize`, `tools/oracle_greedy` |

### Install

```sh
# hf CLI (once): uv tool install huggingface_hub
export PATH="$HOME/.local/bin:$PATH"
hf download shibatch/stories-converted stories15M.F16.gguf --local-dir models
```

### Smoke

```sh
./jllama_cli.ijs -m models/stories15M.F16.gguf -p "Once upon a time" -n 32
```

Expected: English-ish story continuation (not fixture gibberish).

### Parity notes

- SPM encode matches libllama for lab prompts (e.g. `Once upon a time` → `1 9038 2501 263 931`).
- Greedy generate matches oracle for a **short prefix** (first ~4 new tokens observed). Longer runs can drift: jllama math is **f64**; llama.cpp is largely f16/f32 kernels.
- Modern chat models (Llama-3.2-1B, etc.) need **GQA** (M13 done for load/forward) and often **quant** (M12) for practical size.

### GQA / larger models (M13)

jllama loads Llama GQA when `n_head_kv` divides `n_head` and attn_k/v widths are `n_head_kv * d_head`. No GQA fixture is shipped; use any F16 Llama GGUF you install under `models/`.

| Requirement | stories15M | Llama-3.2-1B class |
|-------------|------------|---------------------|
| `general.architecture=llama` | yes | yes |
| F16 GGUF | yes | sometimes |
| Attention | MHA | GQA (supported M13) |
| Size on M2 32GB as J f64 | ~0.12 GB weights | ~8 GB weights |
| Quant | not needed | often needed (M12) |

## Optional GQA lab (M13): Llama-3.2-1B Instruct F16

| Field | Value |
|-------|--------|
| Path | `models/Llama-3.2-1B-Instruct-f16.gguf` (gitignored, ~2.3 GB) |
| Source | [bartowski/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) |
| Hparams | n_vocab=128256, n_embd=2048, n_head=32, n_head_kv=8, n_layer=16, n_ff=8192, θ=500000 |
| RAM | ~5–8 GB as J f64 after load |

```sh
export PATH="$HOME/.local/bin:$PATH"
hf download bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-f16.gguf --local-dir models
./jllama_cli.ijs -m models/Llama-3.2-1B-Instruct-f16.gguf -p "The capital of France is" -n 16
```

Notes: load is F16→f64 (LUT decode); pure-J generate is slow vs llama.cpp; **BOS is prepended** (`pre=llama-bpe`); no chat template (Instruct models still need manual headers for chat); no quant.

## Lab / synthetic

| Name | Notes |
|------|--------|
| `make_synthetic` | Deterministic tiny weights — M2–M3 |
| `model_from_gguf` | Llama GGUF F16/F32 MHA/GQA — M4+M13 |
