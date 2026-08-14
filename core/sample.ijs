NB. jllama sampling (M7)
NB.
NB. Locale: jllamasample
NB.
NB. Pipeline (llama.cpp-ish):
NB.   logits -> / temp (if temp>0) -> top-k filter -> softmax -> top-p filter -> sample
NB.   temp <= 0  => greedy argmax (top-k/top-p ignored)
NB.   top_k <= 0 => disabled
NB.   top_p >= 1 => disabled; top_p <= 0 treated as disabled
NB.
NB. Sample cfg (open numeric list or scalar box of same):
NB.   temp ; top_k ; top_p ; seed ; eos_id ; stop_on_eos
NB. Defaults: 0 ; 0 ; 1 ; 0 ; _1 ; 1
NB.   eos_id < 0  => no EOS id
NB.   stop_on_eos = 0/1
NB.
NB. Public:
NB.   default_cfg ''
NB.   sample_cfg_pack y     normalize open/boxed cfg
NB.   sample_next cfg ; logits  -> (<token) , (<new_seed)
NB.   sample_greedy logits
NB.   rng_u01 seed -> (<u) , (<seed2)

cocurrent 'jllamasample'

NB. Tensor helpers into this locale (softmax, …)
load jpath '~temp/jllama/core/tensor.ijs'
MASK =: _1e30

NB. ---------------------------------------------------------------
NB. Config
NB. ---------------------------------------------------------------
NB. open numeric: temp top_k top_p seed eos_id stop_on_eos
default_cfg =: 3 : '0 0 1 0 _1 1'

NB. accept open list, ;-list of boxes, or scalar box; fill defaults
sample_cfg_pack =: 3 : 0
  c =. y
  if. 0 = # , c do. default_cfg '' return. end.
  if. 32 = 3!:0 c do.
    NB. scalar box of list, or list of boxes from ;
    if. 0 = #$ c do. c =. > c end.
    if. 32 = 3!:0 c do. c =. > c end.
  end.
  c =. , c
  d =. default_cfg ''
  n =. # c
  if. n < # d do. c =. c , n }. d end.
  if. n > # d do. c =. (# d) {. c end.
  c
)

NB. ---------------------------------------------------------------
NB. RNG (LCG, deterministic)
NB. seed is integer; returns u in [0,1) and next seed
NB. ---------------------------------------------------------------
rng_u01 =: 3 : 0
  s =. <. y
  NB. Numerical Recipes LCG mod 2^31
  s =. 2147483647 (17 b.) 1664525 * s + 1013904223
  if. s < 0 do. s =. s + 2147483647 end.
  u =. s % 2147483647
  (<u) , (<s)
)

NB. ---------------------------------------------------------------
NB. Filters
NB. ---------------------------------------------------------------

NB. greedy argmax (first max on ties)
sample_greedy =: 3 : '(i. >./) , y'

NB. k top_k_filter logits -> logits with non-top-k set to MASK
NB. k<=0 => no-op
top_k_filter =: 4 : 0
  k =. x
  z =. , y
  n =. # z
  if. (k <: 0) +. k >: n do. z return. end.
  NB. indices of top-k by value (stable-ish via grade)
  ord =. \: z
  keep =. k {. ord
  out =. n $ MASK
  out =. (keep { z) keep } out
  out
)

NB. p top_p_filter probs -> renormalized probs (mass outside nucleus = 0)
NB. Sort descending, keep smallest set with cumprob >= p (at least 1).
top_p_filter =: 4 : 0
  p =. x
  pr =. , y
  n =. # pr
  if. (p >: 1) +. p <: 0 do. pr return. end.
  ord =. \: pr
  srt =. ord { pr
  c =. +/\ srt
  NB. keep while previous cum < p; always keep first
  keepn =. 1 + +/ }: c < p
  keepn =. n <. 1 >. keepn
  keep =. keepn {. ord
  out =. n $ 0
  out =. (keep { pr) keep } out
  s =. +/ out
  if. s > 0 do. out % s else. pr end.
)

NB. sample index from probability vector with u in [0,1)
NB. categorical via CDF
sample_from_probs =: 4 : 0
  u =. x
  pr =. , y
  c =. +/\ pr
  NB. first index where cdf >= u; if u is 0, first positive mass
  if. u <: 0 do.
    (pr > 0) i. 1 return.
  end.
  i =. (c >: u) i. 1
  if. i >: # c do. i =. <: # c end.
  i
)

NB. ---------------------------------------------------------------
NB. sample_next: cfg ; logits -> (<tok) , (<seed2)
NB. ---------------------------------------------------------------
sample_next =: 3 : 0
  NB. y = (<cfg) , (<logits)   or   cfg ; logits (cfg open numeric)
  if. 2 = # y do.
    cfg =. 0 { y
    logits =. 1 { y
    if. 32 = 3!:0 cfg do. cfg =. > cfg end.
    if. 32 = 3!:0 logits do. logits =. > logits end.
  else.
    'sample_next: want cfg ; logits or (<cfg),(<logits)' assert 0
  end.
  cfg =. sample_cfg_pack cfg
  'temp top_k top_p seed eos_id stop_on_eos' =. cfg
  z =. , logits
  if. temp <: 0 do.
    tok =. sample_greedy z
    (<tok) , (<seed) return.
  end.
  z =. z % temp
  z =. top_k top_k_filter z
  pr =. softmax z
  pr =. top_p top_p_filter pr
  'u seed2' =. rng_u01 seed
  tok =. u sample_from_probs pr
  (<tok) , (<seed2)
)

cocurrent 'base'
