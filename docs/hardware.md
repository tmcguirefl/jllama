# Hardware notes - test model sizing

**Machine:** MacBook Pro, Apple M2, **32 GB** unified memory  
**Runtime:** J 9.8 jconsole, arm64 Darwin  
**v1 weights in RAM:** J **float64** (F16/F32 GGUF tensors promoted on load)

## Why J needs more RAM than llama.cpp for the same GGUF

| Engine | Typical weight storage | 1B dense params (approx) |
|--------|------------------------|---------------------------|
| llama.cpp Q4_K_M | ~0.6 GB | fits easy |
| llama.cpp F16 | ~2.0 GB | easy |
| jllama v1 (F16 file -> J f64) | ~8 bytes/param | **~8 GB** weights alone |
| Activations + KV + interpreter overhead | extra | often +1-4 GB depending on ctx |

Rule of thumb for jllama v1:

```text
weight_RAM_GiB ~ 8 * (params_billions)
comfortable_total ~ weights + KV + ~2-4 GiB headroom for J/OS
```

On **32 GB**, leave room for macOS + browser + jconsole. Budget **~20-24 GB** peak for jllama experiments.

## Recommendations

### Primary test target (M4-M6 parity work)

| Choice | Params | F16 file | J f64 weights | Verdict |
|--------|--------|----------|---------------|---------|
| **Best default** | **~0.5B-1B** | ~1-2 GB | **~4-8 GB** | Sweet spot: real model, parity meaningful, RAM OK |
| Stretch | ~1.5B-3B | ~3-6 GB | ~12-24 GB | Possible at short ctx; little headroom |
| Too big for v1 F64 | 7B F16 | ~14 GB | **~56 GB** | Does not fit; needs quant path or external engine |

Concrete HF/gguf-class examples to hunt (names change; pick any **Llama-arch dense** GGUF in **F16**):

- **~0.5B-1B Instruct F16** - preferred M6 oracle partner  
- Tiny proxies (**100M-360M**) - faster iteration for M1-M3 shape tests and smoke generate  

Examples of size *classes* (verify arch + file dtype before relying on them):

- `Llama-3.2-1B` (or similar 1B) **F16** GGUF - primary candidate  
- `Llama-3.2-3B` F16 - only after 1B is green; watch RAM  
- Very small Llama-compatible or random-weight tensors for unit tests - no download required (M2-M3)

Prefer **F16** over Q4 for v1 so we do not debug dequant and math at once. llama.cpp remains the oracle on the **same** F16 file (or on Q4 only after M9a dequant).

### Context length while testing

| ctx | Use |
|-----|-----|
| 128-512 | Primitive + block unit tests |
| 512-2048 | Interactive generate / parity prompts |
| 4k+ | Later; KV grows with `n_layer * n_kv_heads * ctx * d_head * 16` bytes (K+V in f64) rough order |

KV is usually smaller than weights at 1B/2k, but still budget it.

### What not to use for early milestones

- **7B+ F16 in pure J f64** on 32 GB  
- Large **MoE** (architecture out of scope)  
- Multimodal projectors  
- "Fast chat" Q4 models as the *first* parity target (use them only after float path works, via dequant)

## llama.cpp on the same Mac

Use llama.cpp freely as oracle; it will run **much** larger models (Q4 7B-70B-class) on this machine. That does **not** mean jllama v1 should target those sizes.

```text
llama.cpp:  good for 7B Q4 chat on M2 32GB
jllama v1:  good for <=1B F16 parity lab (<=3B stretch)
```

## Decision for this project

| Role | Size |
|------|------|
| Unit tests (M1-M3) | Synthetic tiny dims + optional <=360M |
| **Integration / parity (M6)** | **~1B F16 Llama-arch GGUF** |
| Stretch demo | ~3B F16 if 1B is solid and RAM allows |
| Deferred | 7B+ via quant dequant (M9) or never in pure f64 |

When a specific GGUF filename is chosen, record it in `docs/models.md` (checksum, `n_layer`, `n_embd`, `n_head`, source URL).
