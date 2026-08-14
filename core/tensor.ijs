NB. jllama core tensor helpers (M1)
NB. No private locale: load this script into the caller's locale.
NB.   cocurrent 'myloc'
NB.   load jpath '~temp/jllama/core/tensor.ijs'
NB.
NB. Defines: silu softmax rmsnorm linear causal_mask allclose
NB.           sftmax make2d RMS_EPS MASK_VAL ATOL RTOL
NB.
NB. Matrix product is not wrapped — use  +/ . *  at call sites.
NB.
NB. Layout (see docs/conventions.md):
NB.   activations: n_tok x n_embd  (or 1D n_embd)
NB.   linear W:    n_in  x n_out   with  x +/ . * w
NB.   softmax:     vector full; table per row (1-cells)

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

NB. Softmax — stable, last-axis for tables.
NB. sftmax: vector (hooks fill both sides of dyads from y)
sftmax =: 13 : '(% +/)(^(- >./)y)'
NB. rank 1 -> sftmax; otherwise sftmax per row
softmax =: sftmax`(sftmax"1)@.(1&~:@#@$)

NB. RMSNorm (Llama-style).
NB. Dyadic:  weight rmsnorm x
NB.   weight: length n_embd
NB.   x:      n_embd  or  n_tok x n_embd
NB.   out:    same shape as x
NB. eps = RMS_EPS (1e_5 default; Llama often 1e-5)
rmsnorm =: 4 : 0
  w =. x
  if. 1 = #$y do.
    ms =. (+/ % #) *: y
    w * y % %: ms + RMS_EPS
  else.
    NB. ms is one scalar per row; divide with rank %"1 0
    ms =. (+/ % #)"1 *: y
    w *"1 y %"1 0 %: ms + RMS_EPS
  end.
)

NB. ---------------------------------------------------------------
NB. Linear + mask
NB. ---------------------------------------------------------------

NB. Affine map.
NB.   linear x;w        -> x +/ . * w
NB.   linear x;w;b      -> b +"1 x +/ . * w
NB. x: n_tok x n_in   w: n_in x n_out   b: n_out
linear =: 3 : 0
  n =. # y
  if. n = 2 do.
    'xv wv' =. y
    xv +/ . * wv
  elseif. n = 3 do.
    'xv wv bv' =. y
    bv +"1 xv +/ . * wv
  elseif. do.
    'linear expects x;w or x;w;b' assert 0
  end.
)

NB. Causal attention mask (n x n).
NB. 0 where key pos <= query pos (allowed); MASK_VAL above diagonal.
NB. Usage:  softmax mask + scores
causal_mask =: MASK_VAL * </~@i.
