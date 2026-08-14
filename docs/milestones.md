# Milestones

jllama reimplements a thin Llama-style decode path in J, validated against llama.cpp where it matters. Product surface is **CLI first**, server later (optional).

## Status board

| ID | Name | Exit criteria | Status |
|----|------|----------------|--------|
| **M0** | Skeleton | `jllama_dev.ijs` loads; smoke + help | **done** |
| **M1** | Tensor lab | rmsnorm, softmax, silu, linear, mask tested | **done** |
| **M2** | Attn + RoPE + KV | cache decode == full recompute (synthetic) | **done** |
| **M3** | Tiny stack | greedy generate on random Llama shapes | **done** |
| **M4** | GGUF F16/F32 | load real file; names/shapes match dump | **done** |
| **M5** | Tokenizer | GPT-2 byte BPE encode/decode from GGUF vocab | **done** |
| **M6** | Parity (fixture) | greedy tokens match llama.cpp on tiny F16 fixture | **done** |
| **M7** | Sampling | temp / top-k / top-p, EOS stop | **done** |
| **M8** | **CLI** | `jllama_cli` shell entry: model, prompt, n, sample flags, print text/ids | **done** |
| **M9** | Performance | profile CLI path; fewer copies; LAPACK/`mp` where useful | **deferred** |
| **M10** | Real-model lab | pin F16 Llama MHA GGUF under `models/`; SPM tokenize; CLI English smoke | **done** |
| **M11** | Server (opt) | thin HTTP/SSE wrapper around same engine as CLI | optional |
| **M12** | Quant (opt) | dequant-to-f64 first (Q4/Q5/Q8), then optional faster paths | optional |
| **M13** | GQA / larger 1B | `n_head_kv` path so Llama-3.2-1B-class models load | **done** |
| **M14** | Second arch (opt) | shared core, second golden model (e.g. Qwen-dense) | optional |

**Completed critical path:** M0 → M8 + **M10** (engine, CLI, real TinyStories lab).  
**Deferred:** M9 performance (may touch J interpreter / LAPACK discussion).  
**Done optional:** M13 GQA. **Still optional:** M11 server, M12 quant, M14 second arch.

## What “done” means for open milestones

### M8 — CLI (**done**)

Thin front-end over existing APIs — not a llama.cpp clone.

```text
./jllama_cli.ijs -m PATH.gguf -p "prompt" -n 64 \
  --temp 0.8 --top-k 40 --top-p 0.95 [--eos ID] [--tokens]
```

Delivered:

- `jllama_cli.ijs` (shebang → jconsole) + locale `jllamacli` (`cli/cli.ijs`)
- Flags: model, prompt (or `-f`), `n_predict`, temp/top-k/top-p/seed, optional EOS, `--tokens`
- Uses `model_from_gguf`, `vocab_from_gguf`, `encode`, `generate_sample`, `decode`
- Works on tiny parity fixtures in the dev tree (greedy ids match M6 oracle)
- **Out of scope (still):** chat templates, REPL conversation, GBNF, HTTP, GPU

### M9 — Performance (**deferred**)

Parked: may involve J interpreter / LAPACK choices beyond app-level tweaks.

- Profile prefill vs decode on CLI path (fixture + stories15M)
- Cut obvious J copies / re-boxes on hot path
- Try `math/lapack2` for large `mp` if win is real
- Record tok/s notes in `docs/`

### M10 — Real-model lab (**done**)

Delivered:

- Primary file: `models/stories15M.F16.gguf` (~15M TinyStories Llama MHA F16) — see [models.md](models.md)
- **Llama SPM tokenizer** in `jllamavocab` (`tokenizer.ggml.model=llama`) alongside GPT-2 BPE
- CLI English continuation: `./jllama_cli.ijs -m models/stories15M.F16.gguf -p "Once upon a time" -n 32`
- Encode/greedy parity vs llama.cpp oracles (dev tree)
- Version **0.10.0**

Not claimed: full long-sequence greedy id match (f64 vs f16 drift), GQA 1B chat models, chat templates.

### M11 — Server (optional)

- Same completion engine as CLI
- Minimal HTTP (e.g. one `/completion` or SSE stream)
- No requirement to match llama-server API 1:1 in v1

### M12 — Quant (optional)

- M12a: GGUF quant tensor → f64, reuse float graph
- Later: faster kernels only if needed

### M13 — GQA / larger 1B (**done**)

Delivered:

- Attention derives `n_head_kv` from Wk/Wv width; expands KV heads (`expand_kv`) at score time
- KV cache stores `n_head_kv` heads (not full `n_head`)
- `hparams` includes trailing `n_head_kv` (legacy 6-item packs still accepted via `hp_open`)
- `model_from_gguf` loads GQA when `llama.attention.head_count_kv` divides `head_count` and K/V widths match
- Synthetic `make_layer` / `make_synthetic` optional `n_head_kv`
- `test/test_m13.ijs` (7 tests): expand, attn/block/stack parity, generate==recompute
- Version **0.13.0**

Not claimed: a pinned ~1B F16 GQA file under `models/` (user supplies Llama-3.2-1B-class F16 when desired; RAM ~8GB f64 weights on M2 32GB). Still no quant (M12).

### M14 — Second arch (optional)

- Shared tensor/attn/sample; arch-specific load + naming

## Design rules (carry forward)

- **J only** in product code (Python OK for fixture writers / labs)
- Locale names: no `_` (`jllamamodel`, not `jllama_model`)
- Nested packs: `(<a),(<b),box` — not chained `;` across box lists
- jconsole: `/Applications/j9.8/bin/jconsole` full path
- Oracle: Homebrew llama.cpp + `tools/oracle_greedy` / `oracle_tokenize`
- Test model sizing: [hardware.md](hardware.md)

## Suggested “what next” order

1. **M12 quant** if you only have Q4/Q5/Q8 files  
2. Pin a ~1B F16 GQA lab model under `models/` (manual download)  
3. **M11 server** if you need a network client  
4. **M9 perf** when ready to profile / discuss interpreter-level work  

## Version mapping

| Version | Milestone |
|---------|-----------|
| 0.1.x | M1 |
| 0.2.x | M2 |
| 0.3.x | M3 |
| 0.4.x | M4 |
| 0.5.x | M5 |
| 0.6.x | M6 |
| 0.7.x | M7 |
| 0.8.x | M8 CLI |
| 0.9.x | M9 perf (deferred) |
| 0.10.x | M10 real-model lab |
| 1.0? | CLI + real-model lab + GQA solid enough to call v1 |
