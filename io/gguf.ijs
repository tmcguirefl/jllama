NB. jllama GGUF F16/F32 loader (M4 + M13 GQA)
NB.
NB. Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
NB. Little-endian GGUF v3. F32 and F16 tensors only (no quant).
NB.
NB. Locale: jllamagguf  (no underscore in locale name)
NB.
NB. Public:
NB.   gguf_load path
NB.     -> <"_ (meta ; tinfos ; align ; data_off ; bytes)
NB.   load gguf_meta key
NB.   load gguf_tensor name   -> J f64 array (jllama layout)
NB.   gguf_names load
NB.   model_from_gguf path    -> jllama model box (Llama dense MHA/GQA)
NB.   gguf_summary path
NB.
NB. Weight layout:
NB.   GGUF/ggml 2d: dims = ne0,ne1 with ne0 contiguous (innermost).
NB.   File ravel -> J  (ne1 , ne0) $ data
NB.   token_embd.weight kept as (n_vocab , n_embd) = (ne1 , ne0)
NB.   other 2d .weight matrices transposed to (ne0 , ne1) = n_in x n_out
NB.   for jllama  x +/ . * w .

cocurrent 'jllamagguf'

DEFAULT_THETA =: DEFAULT_THETA_jllamarope_

NB. ---------------------------------------------------------------
NB. Constants
NB. ---------------------------------------------------------------
NB. Disk bytes "GGUF" = 47 47 55 46; LE u32 value:
GGUF_MAGIC =: 16b46554747
GGUF_VERSION =: 3
ALIGN_DEFAULT =: 32

GGML_F32 =: 0
GGML_F16 =: 1

VT_UINT8 =: 0
VT_INT8 =: 1
VT_UINT16 =: 2
VT_INT16 =: 3
VT_UINT32 =: 4
VT_INT32 =: 5
VT_FLOAT32 =: 6
VT_BOOL =: 7
VT_STRING =: 8
VT_ARRAY =: 9
VT_UINT64 =: 10
VT_INT64 =: 11
VT_FLOAT64 =: 12

NB. ---------------------------------------------------------------
NB. Byte reader state (one load at a time)
NB. ---------------------------------------------------------------
rd_init =: 3 : 0
  RD_BYTES =: y
  RD_OFF =: 0
  RD_N =: # y
  i.0 0
)

rd_need =: 3 : 0
  if. (RD_OFF + y) > RD_N do.
    'gguf: unexpected EOF' assert 0
  end.
  i.0 0
)

rd_take =: 3 : 0
  n =. y
  rd_need n
  o =. RD_OFF
  RD_OFF =: o + n
  n {. o }. RD_BYTES
)

align_up =: 4 : 0
  a =. x
  o =. y
  o + a | - o
)

NB. ---------------------------------------------------------------
NB. LE decoders (char vector -> numbers)
NB. J 3!:4 type codes (dyadic): 1=16-bit, 2=32-bit, 3=64-bit int
NB. Negative left arg decodes. fc: _1=f32, _2=f64.
NB. ---------------------------------------------------------------
u32 =: 3 : '_2 (3!:4) y'
u64 =: 3 : '_3 (3!:4) y'
f32 =: 3 : '_1 fc y'
f64 =: 3 : '_2 fc y'

NB. u16 LE bytes -> unsigned ints
u16 =: 3 : 0
  (_1 (3!:4) y) (17 b.) 16bffff
)

NB. IEEE binary16 bits (u16) -> J floats (vectorized).
NB. Avoid b. shifts (opcode confusion); use integer divide / mask.
NB. Scalar loop was ~2.5e6 elems/s (~8+ min for 1B F16); this is array-wide.
f16_from_bits_raw =: 3 : 0
  u =. , y
  sign =. _1 ^ u >: 16b8000
  exp =. 16b1f (17 b.) <. u % 1024
  mant =. 16b3ff (17 b.) u
  frac =. mant % 1024
  NB. normal: (1 + m/1024) * 2^(e-15)
  v =. (1 + frac) * 2 ^ (exp - 15)
  NB. subnormal / zero: e=0
  sub =. exp = 0
  v =. (sub * frac * 2 ^ _14) + ((-. sub) * v)
  NB. inf / nan: e=31
  hi =. exp = 31
  if. +./ hi do.
    v =. _ (I. hi *. mant = 0)} v
    v =. _. (I. hi *. mant ~: 0)} v
  end.
  sign * v
)

NB. 65536-entry LUT (built once at load). Indexing beats re-decode on big tensors.
F16_LUT =: f16_from_bits_raw i. 65536

NB. y = u16 ints (any shape); ravel-index via LUT
f16_from_bits =: 3 : 0
  F16_LUT {~ 16bffff (17 b.) , y
)

f16_bytes =: 3 : 0
  b =. y
  'f16_bytes: odd length' assert 0 = 2 | # b
  f16_from_bits u16 b
)

rd_u32 =: 3 : 'u32 rd_take 4'
rd_u64 =: 3 : 'u64 rd_take 8'
rd_f32 =: 3 : 'f32 rd_take 4'
rd_f64 =: 3 : 'f64 rd_take 8'
rd_u8 =: 3 : 'a. i. rd_take 1'
rd_bool =: 3 : '0 ~: rd_u8 '''''

rd_string =: 3 : 0
  n =. rd_u64 ''
  if. n = 0 do. '' return. end.
  rd_take n
)

rd_i8 =: 3 : 0
  v =. rd_u8 ''
  if. v > 127 do. v =. v - 256 end.
  v
)

rd_u16 =: 3 : 'u16 rd_take 2'

rd_i16 =: 3 : 0
  v =. rd_u16 ''
  if. v > 32767 do. v =. v - 65536 end.
  v
)

rd_array =: 3 : 0
  at =. rd_u32 ''
  n =. rd_u64 ''
  if. n = 0 do.
    if. at = VT_STRING do. 0 $ a: else. 0 $ 0 end.
    return.
  end.
  if. at = VT_STRING do.
    r =. 0 $ a:
    for. i. n do. r =. r , < rd_string '' end.
    r return.
  end.
  if. at = VT_ARRAY do.
    r =. 0 $ a:
    for. i. n do. r =. r , < rd_array '' end.
    r return.
  end.
  r =. ''
  for. i. n do.
    if. at = VT_UINT8 do. v =. rd_u8 ''
    elseif. at = VT_INT8 do. v =. rd_i8 ''
    elseif. at = VT_UINT16 do. v =. rd_u16 ''
    elseif. at = VT_INT16 do. v =. rd_i16 ''
    elseif. at = VT_UINT32 do. v =. rd_u32 ''
    elseif. at = VT_INT32 do. v =. rd_u32 ''
    elseif. at = VT_FLOAT32 do. v =. rd_f32 ''
    elseif. at = VT_BOOL do. v =. rd_bool ''
    elseif. at = VT_UINT64 do. v =. rd_u64 ''
    elseif. at = VT_INT64 do. v =. rd_u64 ''
    elseif. at = VT_FLOAT64 do. v =. rd_f64 ''
    elseif. do. 'gguf: bad array elem type' assert 0
    end.
    r =. r , v
  end.
  r
)

rd_value =: 3 : 0
  t =. rd_u32 ''
  if. t = VT_UINT8 do. rd_u8 '' return. end.
  if. t = VT_INT8 do. rd_i8 '' return. end.
  if. t = VT_UINT16 do. rd_u16 '' return. end.
  if. t = VT_INT16 do. rd_i16 '' return. end.
  if. t = VT_UINT32 do. rd_u32 '' return. end.
  if. t = VT_INT32 do. rd_u32 '' return. end.
  if. t = VT_FLOAT32 do. rd_f32 '' return. end.
  if. t = VT_BOOL do. rd_bool '' return. end.
  if. t = VT_STRING do. rd_string '' return. end.
  if. t = VT_ARRAY do. rd_array '' return. end.
  if. t = VT_UINT64 do. rd_u64 '' return. end.
  if. t = VT_INT64 do. rd_u64 '' return. end.
  if. t = VT_FLOAT64 do. rd_f64 '' return. end.
  smoutput 'gguf: bad value type ' , ": t
  'gguf: unsupported metadata type' assert 0
)

NB. ---------------------------------------------------------------
NB. File load
NB. ---------------------------------------------------------------

NB. y = path
gguf_load =: 3 : 0
  path =. y
  if. -. fexist path do.
    smoutput 'gguf: missing ' , path
    'gguf: file not found' assert 0
  end.
  bytes =. 1!:1 < path
  rd_init bytes
  magic =. rd_u32 ''
  'gguf: bad magic' assert magic = GGUF_MAGIC
  ver =. rd_u32 ''
  'gguf: unsupported version (want 3)' assert ver = GGUF_VERSION
  n_tensors =. rd_u64 ''
  n_kv =. rd_u64 ''
  meta =. 0 $ a:
  for. i. n_kv do.
    k =. rd_string ''
    v =. rd_value ''
    meta =. meta , < (<k) , (<v)
  end.
  align =. ALIGN_DEFAULT
  m =. meta gguf_meta_get_from 'general.alignment'
  if. # m do. align =. > 0 { m end.
  tinfos =. 0 $ a:
  for. i. n_tensors do.
    name =. rd_string ''
    nd =. rd_u32 ''
    'gguf: bad ndim' assert (nd >: 1) *. nd <: 4
    dims =. ''
    for. i. nd do. dims =. dims , rd_u64 '' end.
    typ =. rd_u32 ''
    off =. rd_u64 ''
    tinfos =. tinfos , < (<name) , (<dims) , (<typ) , (<off)
  end.
  data_off =. align align_up RD_OFF
  <"_ (meta ; tinfos ; align ; data_off ; bytes)
)

NB. x = meta list ; y = key -> 0$0 or ,<value
gguf_meta_get_from =: 4 : 0
  meta =. x
  key =. y
  for_m. meta do.
    'k v' =. > m
    if. key -: k do. , < v return. end.
  end.
  0 $ 0
)

NB. load gguf_meta key
gguf_meta =: 4 : 0
  'meta tinfos align data_off bytes' =. > x
  r =. meta gguf_meta_get_from y
  if. 0 = # r do.
    'gguf: missing metadata key ' , y assert 0
  end.
  > 0 { r
)

NB. load gguf_meta_default (key ; default)
gguf_meta_default =: 4 : 0
  'meta tinfos align data_off bytes' =. > x
  'key def' =. y
  r =. meta gguf_meta_get_from key
  if. 0 = # r do. def else. > 0 { r end.
)

gguf_names =: 3 : 0
  'meta tinfos align data_off bytes' =. > y
  r =. 0 $ a:
  for_t. tinfos do.
    'name dims typ off' =. > t
    r =. r , < name
  end.
  r
)

NB. load gguf_find name -> info box or empty
gguf_find =: 4 : 0
  'meta tinfos align data_off bytes' =. > x
  want =. y
  for_t. tinfos do.
    'name dims typ off' =. > t
    if. want -: name do. t return. end.
  end.
  0 $ 0
)

elem_nbytes =: 3 : 0
  if. y = GGML_F32 do. 4 return. end.
  if. y = GGML_F16 do. 2 return. end.
  'gguf: only F16/F32 tensors supported' assert 0
)

NB. load gguf_tensor name -> numeric array
gguf_tensor =: 4 : 0
  load =. x
  want =. y
  info =. load gguf_find want
  'gguf: missing tensor ' , want assert 0 ~: # info
  'name dims typ off' =. > info
  'meta tinfos align data_off bytes' =. > load
  nb =. elem_nbytes typ
  n_elem =. */ dims
  nbytes =. n_elem * nb
  start =. data_off + off
  'gguf: tensor OOB' assert (start + nbytes) <: # bytes
  chunk =. nbytes {. start }. bytes
  if. typ = GGML_F32 do.
    data =. f32 chunk
  else.
    data =. f16_bytes chunk
  end.
  'gguf: bad element count' assert n_elem = # , data
  data =. , data
  nd =. # dims
  if. nd = 1 do.
    data return.
  end.
  if. nd = 2 do.
    ne0 =. 0 { dims
    ne1 =. 1 { dims
    raw =. (ne1 , ne0) $ data
    if. name -: 'token_embd.weight' do.
      raw return.
    end.
    |: raw return.
  end.
  (|. dims) $ data
)

NB. ---------------------------------------------------------------
NB. Llama-arch -> jllama model box
NB. ---------------------------------------------------------------
model_from_gguf =: 3 : 0
  load =. gguf_load y
  arch =. load gguf_meta 'general.architecture'
  'model_from_gguf: only llama arch for M4' assert arch -: 'llama'
  NB. {. forces true scalars (GGUF meta numbers are often length-1 lists)
  n_embd =. {. load gguf_meta 'llama.embedding_length'
  n_layer =. {. load gguf_meta 'llama.block_count'
  n_ff =. {. load gguf_meta 'llama.feed_forward_length'
  n_head =. {. load gguf_meta 'llama.attention.head_count'
  n_head_kv =. {. load gguf_meta_default 'llama.attention.head_count_kv' ; n_head
  'model_from_gguf: n_head_kv must divide n_head' assert 0 = n_head_kv | n_head
  'model_from_gguf: n_head_kv > 0' assert n_head_kv > 0
  'model_from_gguf: n_embd not divisible by n_head' assert 0 = n_head | n_embd
  d_head =. n_embd % n_head
  theta =. {. load gguf_meta_default 'llama.rope.freq_base' ; DEFAULT_THETA
  wte =. load gguf_tensor 'token_embd.weight'
  n_vocab =. # wte
  'model_from_gguf: bad embd width' assert n_embd = {: $ wte
  vs =. load gguf_meta_default 'llama.vocab_size' ; n_vocab
  'model_from_gguf: vocab_size mismatch' assert vs = n_vocab
  ln_f =. load gguf_tensor 'output_norm.weight'
  if. 0 ~: # load gguf_find 'output.weight' do.
    lm_head =. load gguf_tensor 'output.weight'
  else.
    lm_head =. |: wte
  end.
  layers =. 0 $ a:
  for_i. i. n_layer do.
    bid =. ": i
    pref =. 'blk.' , bid , '.'
    attn_n =. load gguf_tensor pref , 'attn_norm.weight'
    wq =. load gguf_tensor pref , 'attn_q.weight'
    wk =. load gguf_tensor pref , 'attn_k.weight'
    wv =. load gguf_tensor pref , 'attn_v.weight'
    'model_from_gguf: bad attn_k width' assert ({: $ wk) = n_head_kv * d_head
    'model_from_gguf: bad attn_v width' assert ({: $ wv) = n_head_kv * d_head
    'model_from_gguf: bad attn_q width' assert ({: $ wq) = n_head * d_head
    wo =. load gguf_tensor pref , 'attn_output.weight'
    ffn_n =. load gguf_tensor pref , 'ffn_norm.weight'
    wg =. load gguf_tensor pref , 'ffn_gate.weight'
    wu =. load gguf_tensor pref , 'ffn_up.weight'
    wd =. load gguf_tensor pref , 'ffn_down.weight'
    layer =. <"_ (attn_n ; wq ; wk ; wv ; wo ; ffn_n ; wg ; wu ; wd)
    layers =. layers , layer
  end.
  hparams =. n_vocab ; n_embd ; n_head ; n_layer ; n_ff ; theta ; n_head_kv
  <"_ (hparams ; wte ; layers ; ln_f ; lm_head)
)

gguf_summary =: 3 : 0
  load =. gguf_load y
  'meta tinfos align data_off bytes' =. > load
  smoutput 'file: ' , y
  smoutput 'bytes: ' , ": # bytes
  smoutput 'align: ' , ": align
  smoutput 'data_off: ' , ": data_off
  smoutput 'n_kv: ' , ": # meta
  smoutput 'n_tensors: ' , ": # tinfos
  arch =. load gguf_meta_default 'general.architecture' ; '(none)'
  smoutput 'arch: ' , arch
  for_t. tinfos do.
    'name dims typ off' =. > t
    smoutput name , '  shape=' , (": dims) , '  type=' , (": typ) , '  off=' , ": off
  end.
  i.0 0
)

cocurrent 'base'
