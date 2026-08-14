NB. jllama RoPE (rotary position embeddings)
NB. Llama / ggml ROPE_TYPE_NORMAL: pair-wise consecutive dims.
NB.   For each pair (x0,x1) at indices 2i, 2i+1 with angle theta_i:
NB.     out0 = x0*cos - x1*sin
NB.     out1 = x0*sin + x1*cos
NB. Equivalent form used here:
NB.   rotate_pairs([x0,x1,x2,x3,...]) = [-x1,x0,-x3,x2,...]
NB.   out = x * cos + rotate_pairs(x) * sin
NB.   with cos/sin layout [c0,c0,c1,c1,...] (ggml NORMAL cache).
NB.
NB. (NeoX / GPT-J half-rotation is NOT used; that is rope_type=2.)
NB.
NB. Last axis is d_head (must be even).
NB. Rank 2: n_tok x d_head
NB. Rank 3: n_tok x n_head x d_head
NB. Positions: integer list length n_tok (usually i. n_tok)
NB.
NB. inv_freq[i] = theta ^ (-(2i)/d) for i = 0 .. d/2-1
NB. e.g. d=4, theta=10000 -> 1 , 0.01
NB.
NB. Load before attention.ijs.

cocurrent 'jllamarope'

DEFAULT_THETA =: 10000

NB. theta rope_inv d_head  -> inv_freq length d_head%2
rope_inv =: 4 : 0
  NB. GGUF meta often yields length-1 lists; ^ needs a true scalar base.
  theta =. {. x
  d =. {. y
  'rope: d_head must be even and positive' assert (d > 0) *. 0 = 2 | d
  % theta ^ ((+: i. -: d) % d)
)

NB. pos rope_sincos inv_freq -> cos ; sin  each (#pos) x d_head
NB. NORMAL layout: duplicate each freq angle across the pair [c0 c0 c1 c1 ...]
rope_sincos =: 4 : 0
  freqs =. x */ y
  c =. 2 o. freqs
  s =. 1 o. freqs
  (2 #"1 c) ; (2 #"1 s)
)

NB. rotate_pairs on last axis: [-x1,x0,-x3,x2,...]
NB. Pairs via _2]\ ; reverse each pair; *"1 multiplies every pair by _1 1.
NB. Parens required so `"1 does not bind across _2.
rotate_pairs1 =: 3 : 0
  'rope: last axis must be even' assert 0 = 2 | # y
  , _1 1 *"1 |."1 (_2 ]\ y)
)
rotate_pairs =: rotate_pairs1"1

NB. Alias kept for callers/tests that still name the helper rotate_half.
NB. Behaviour is NORMAL pair rotation, not NeoX half-rotation.
rotate_half =: rotate_pairs

NB. n_head expand_heads (n_tok x d) -> n_tok x n_head x d
NB. J does not broadcast (n_tok x 1 x d) against (n_tok x n_head x d).
NB. Copy each position row across heads (x # y), then reshape.
expand_heads =: 4 : 0
  'n_tok d' =. $ y
  (n_tok , x , d) $ , x # y
)

NB. Monadic: positions i. n_tok , default theta
NB. Dyadic:  pos rope x   or   (pos;theta) rope x
rope =: 3 : 0
  (i. {. $ y) rope y
:
  if. 32 = 3!:0 x do.
    'pos theta' =. 2 {. x , < DEFAULT_THETA
  else.
    pos =. x
    theta =. DEFAULT_THETA
  end.
  rnk =. #$ y
  'rope: expected rank 2 or 3' assert rnk e. 2 3
  d =. {: $ y
  inv =. theta rope_inv d
  'cs ss' =. pos rope_sincos inv
  if. rnk = 3 do.
    nh =. 1 { $ y
    cs =. nh expand_heads cs
    ss =. nh expand_heads ss
  end.
  (y * cs) + (rotate_pairs y) * ss
)

cocurrent 'base'
