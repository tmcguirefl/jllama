NB. jllama core tensor helpers (M1)
NB. No private locale: load this script into the caller's locale.
NB.   cocurrent 'myloc'
NB.   load jpath '~temp/jllama/core/tensor.ijs'
NB.
NB. Defines: silu softmax rmsnorm linear causal_mask allclose
NB.           make2d RMS_EPS MASK_VAL ATOL RTOL
NB.
NB. Matrix product is not wrapped — use  +/ . *  at call sites.
NB.
NB. Layout:
NB.   activations: n_tok x n_embd  (or 1D n_embd)
NB.   linear W:    n_in  x n_out   with  x +/ . * w
NB.   softmax:     per 1-cell (vector full; table per row)

NB. ---------------------------------------------------------------
NB. Constants
NB. ---------------------------------------------------------------
RMS_EPS =: 1e_5
MASK_VAL =: _1e10           NB. added to forbidden attention scores
ATOL =: 1e_9
RTOL =: 1e_6

NB. ---------------------------------------------------------------
NB. Helpers
NB. ---------------------------------------------------------------

NB. Ensure rank >= 2 (table). Vectors become 1 x n.
make2d =: ,:^:(1 = #@$)

NB. Numeric closeness: 1 iff all atoms match within atol/rtol.
NB. x allclose y
allclose =: 4 : 0
  *./ , (| x - y) <: ATOL + RTOL * | y
)

NB. ---------------------------------------------------------------
NB. Elementwise / reductions
NB. ---------------------------------------------------------------

NB. SiLU / swish: x * sigmoid(x) = x % (1 + exp(-x))
silu =: ] % 1 + ^@:-

NB. Softmax — stable, per 1-cell (picoGPT-in-J style).
NB. Works for vectors and tables (including 1 x N).
softmax =: {{ (% +/) ^ (- >./) y }}"1

NB. RMSNorm (Llama-style), per 1-cell — same pattern as softmax"1.
NB. Dyadic:  weight rmsnorm activations
NB.   x (left):  weight/gain vector, length n_embd
NB.              (ggml/llama.cpp: attn_norm / ffn_norm / output_norm .weight)
NB.   y (right): n_embd  or  n_tok x n_embd
NB.   out:       same shape as y
NB. eps = RMS_EPS (1e_5; Llama often 1e-5)
rmsnorm =: {{
  NB. x = RMSNorm weight (gain), length n_embd — not reassigned; used as left arg
  ms =. (+/ % #) *: y
  x * y % %: ms + RMS_EPS
}}"1

NB. ---------------------------------------------------------------
NB. Linear + mask
NB. ---------------------------------------------------------------

NB. Affine map: always x ; w ; b  (no-bias callers pass b = 0).
NB.   linear x;w;b  ->  b +"1 x +/ . * w
NB. x: n_tok x n_in   w: n_in x n_out   b: n_out (or scalar 0)
linear =: 3 : 0
  'linear expects x;w;b' assert 3 = # y
  'xv wv bv' =. y
  bv +"1 xv +/ . * wv
)

NB. Causal attention mask (n x n).
NB. 0 where key pos <= query pos (allowed); MASK_VAL above diagonal.
NB. Usage:  softmax mask + scores
causal_mask =: MASK_VAL * </~@i.
