# core/

Array ops and transformer math.

| File | Milestone | Status |
|------|-----------|--------|
| `tensor.ijs` | M0-M1 | done |
| `rope.ijs` | M2 | done |
| `attention.ijs` | M2+M13 | done - MHA/GQA + KV |
| `block.ijs` | M3+M13 | done - SwiGLU + pre-norm block (GQA) |
| `model.ijs` | M3+M13 | done - stack + generate + GQA hparams |
| `sample.ijs` | M7 | **done** - temp / top-k / top-p / EOS |

## Modules / locales

### core/tensor.ijs (M1) — no private locale

Load into the caller’s locale (`load jpath '~temp/jllama/core/tensor.ijs'` after `./publish_jllama`).

`silu` `softmax` `rmsnorm` `linear` `causal_mask` `allclose`

Matrix product is not wrapped — use `+/ . *` at call sites.

### jllamarope (M2)

`rope` `rotate_pairs` (`rotate_half` alias) — Llama NORMAL pair-wise RoPE

### jllamaattn (M2+M13)

`mha_full` `mha_step` `mha_prefill_cached` `kv_empty` `expand_kv` `project_qkv`

GQA: Wk/Wv may be narrower; cache is `n_past x n_head_kv x d_head`.

### jllamablock (M3)

| Verb | Role |
|------|------|
| `ffn_swiglu` | SiLU(gate)*up then down |
| `block_full` | full-seq pre-norm block |
| `block_step` | one token + layer KV |
| `block_prefill_cached` | prefill via steps |

Layer (one scalar box):
`<"_ (attn_norm ; wq ; wk ; wv ; wo ; ffn_norm ; w_gate ; w_up ; w_down)`

Call packs use `,` of scalar boxes, e.g.
`block_full (<x) , (<n_head) , layer [ , (<theta) ]`

### jllamamodel (M3)

| Verb | Role |
|------|------|
| `make_synthetic` | tiny deterministic model |
| `forward_full` | no-cache stack |
| `forward_prefill` / `forward_step` | KV path |
| `generate` | greedy with cache (`temp=0`) |
| `generate_sample` | temp/top-k/top-p + EOS stop |
| `generate_fullrecompute` | slow greedy oracle |

### jllamasample (M7)

| Verb | Role |
|------|------|
| `sample_next` | `(<cfg),(<logits)` -> `(<tok),(<seed2)` |
| `top_k_filter` / `top_p_filter` | nucleus filters |
| `default_cfg` | `0 0 1 0 _1 1` |

cfg: `temp ; top_k ; top_p ; seed ; eos_id ; stop_on_eos` (`temp<=0` => greedy).

```j
cfg =. 0.8 ; 40 ; 0.95 ; 1 ; _1 ; 1
m generate_sample_jllamamodel_ (<ids) , (<n_new) , (<cfg)
```

Model (one scalar box):
`<"_ (hparams ; wte ; layers ; ln_f ; lm_head)`  
hparams: `n_vocab; n_embd; n_head; n_layer; n_ff; theta; n_head_kv`

Nested args use `(<a),(<b),box` — not chained `;` across box lists (chained `;` re-boxes left box-lists).

Run tests: `jllama_test ''`
