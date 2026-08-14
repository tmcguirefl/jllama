NB. jllama tokenizer (M5 + M10 + Llama-3 BPE BOS)
NB.
NB. Supports:
NB.   gpt2|bpe  - GPT-2 byte-level BPE (fixtures; Llama-3 uses model=gpt2 pre=llama-bpe)
NB.   llama     - SentencePiece-style SPM (llama.cpp llm_tokenizer_spm)
NB. Llama-3: missing add_bos_token + pre=llama-bpe => add_bos=1 (match libllama)
NB.
NB. Locale: jllamavocab
NB.
NB. Vocab box (scalar):
NB.   <"_ ((<model),(<tokens),(<ranks),(<bos),(<eos),(<unk),
NB.        (<add_bos),(<add_eos),(<pre),(<scores),(<add_sp))
NB. tokens = boxed token strings (id = index)
NB. ranks  = BPE merges only (SPM: empty)
NB. scores = SPM piece scores (BPE: empty)
NB. add_sp = SPM leading space-prefix before escape (default 1 for llama)
NB.
NB. Public:
NB.   vocab_from_gguf path
NB.   vocab_from_load loadbox
NB.   encode  vocab encode text  -> int ids
NB.   decode  vocab decode ids   -> text
NB.   vocab_token / vocab_bos / vocab_eos / vocab_unk / vocab_tokens

cocurrent 'jllamavocab'

NB. U+2581 lower one eighth block (SPM space marker), as unicode char
SPM_WS =: u: 16b2581

NB. ---------------------------------------------------------------
NB. GPT-2 bytes_to_unicode
NB. ---------------------------------------------------------------

make_byte_maps =: 3 : 0
  bs =. (33 + i. 94) , (161 + i. 12) , (174 + i. 82)
  cs =. bs
  n =. 0
  for_b. i. 256 do.
    if. -. b e. bs do.
      bs =. bs , b
      cs =. cs , 256 + n
      n =. n + 1
    end.
  end.
  u =. 256 $ 0
  for_i. i. # bs do.
    u =. (i { cs) (i { bs) } u
  end.
  r =. 400 $ _1
  for_b. i. 256 do.
    r =. b (b { u) } r
  end.
  r ; u
)

'BYTE_RMAP BYTE_UCP' =: make_byte_maps ''

NB. ---------------------------------------------------------------
NB. UTF-8
NB. ---------------------------------------------------------------

utf8_cps =: 3 : 0
  b =. a. i. , y
  if. 0 = # b do. 0 $ 0 return. end.
  out =. 0 $ 0
  i =. 0
  n =. # b
  while. i < n do.
    c =. i { b
    if. c < 128 do.
      out =. out , c
      i =. i + 1
    elseif. c < 224 do.
      'vocab: truncated utf8' assert (i + 1) < n
      c2 =. (i + 1) { b
      out =. out , (64 * 31 (17 b.) c) + 63 (17 b.) c2
      i =. i + 2
    elseif. c < 240 do.
      'vocab: truncated utf8' assert (i + 2) < n
      c2 =. (i + 1) { b
      c3 =. (i + 2) { b
      out =. out , (4096 * 15 (17 b.) c) + (64 * 63 (17 b.) c2) + 63 (17 b.) c3
      i =. i + 3
    elseif. do.
      'vocab: truncated utf8' assert (i + 3) < n
      c2 =. (i + 1) { b
      c3 =. (i + 2) { b
      c4 =. (i + 3) { b
      out =. out , (262144 * 7 (17 b.) c) + (4096 * 63 (17 b.) c2) + (64 * 63 (17 b.) c3) + 63 (17 b.) c4
      i =. i + 4
    end.
  end.
  out
)

utf8_decode =: 3 : 0
  if. 0 = # y do. '' return. end.
  if. 131072 = 3!:0 y do. y return. end.
  u: utf8_cps y
)

utf8_bytes =: 3 : 0
  if. 0 = # y do. '' return. end.
  if. 2 = 3!:0 y do.
    NB. already 8-bit char vector
    if. *./ 255 >: a. i. y do. y return. end.
  end.
  cps =. 3 u: y
  out =. ''
  for_c. cps do.
    if. c < 128 do.
      out =. out , c { a.
    elseif. c < 2048 do.
      out =. out , (192 + <. c % 64) { a.
      out =. out , (128 + 63 (17 b.) c) { a.
    elseif. c < 65536 do.
      out =. out , (224 + <. c % 4096) { a.
      out =. out , (128 + 63 (17 b.) <. c % 64) { a.
      out =. out , (128 + 63 (17 b.) c) { a.
    elseif. do.
      out =. out , (240 + <. c % 262144) { a.
      out =. out , (128 + 63 (17 b.) <. c % 4096) { a.
      out =. out , (128 + 63 (17 b.) <. c % 64) { a.
      out =. out , (128 + 63 (17 b.) c) { a.
    end.
  end.
  out
)

bytes_to_bpe_chars =: 3 : 0
  b =. a. i. , y
  u: b { BYTE_UCP
)

bpe_chars_to_bytes =: 3 : 0
  cps =. 3 u: y
  b =. 0 $ 0
  for_c. cps do.
    if. (c >: 0) *. c < # BYTE_RMAP do.
      bb =. c { BYTE_RMAP
    else.
      bb =. _1
    end.
    'vocab: bad bpe char in decode' assert bb >: 0
    b =. b , bb
  end.
  b
)

bytes_to_text =: 3 : 0
  b =. , y
  if. 0 = # b do. '' return. end.
  utf8_decode b { a.
)

NB. ---------------------------------------------------------------
NB. BPE core (GPT-2)
NB. ---------------------------------------------------------------

bpe_merge =: 4 : 0
  ranks =. x
  word =. y
  if. 2 > # word do. word return. end.
  while. 1 do.
    best_r =. _
    best_i =. _1
    n =. # word
    i =. 0
    while. i < n - 1 do.
      pair =. (< > i { word) , (< > (i + 1) { word)
      r =. _1
      j =. 0
      while. j < # ranks do.
        if. pair -: > j { ranks do. r =. j break. end.
        j =. j + 1
      end.
      if. (r >: 0) *. r < best_r do.
        best_r =. r
        best_i =. i
      end.
      i =. i + 1
    end.
    if. best_i < 0 do. break. end.
    left =. > best_i { word
    right =. > (best_i + 1) { word
    merged =. left , right
    word =. (best_i {. word) , (<merged) , (best_i + 2) }. word
  end.
  word
)

NB. ---------------------------------------------------------------
NB. Vocab load
NB. ---------------------------------------------------------------

tok_eq =: 4 : 0
  NB. Compare as codepoints when possible; fall back to raw bytes.
  try.
    (3 u: x) -: (3 u: y)
  catch.
    x -: y
  end.
)

token_to_id =: 4 : 0
  tokens =. x
  want =. y
  tokens i. < want
)

NB. y = gguf load box
vocab_from_load =: 3 : 0
  load =. y
  model =. load gguf_meta_jllamagguf_ 'tokenizer.ggml.model'
  ok =. +./ model&-: &> 'gpt2' ; 'bpe' ; 'llama'
  'vocab: unsupported tokenizer model (want gpt2|bpe|llama)' assert ok
  raw_tokens =. load gguf_meta_jllamagguf_ 'tokenizer.ggml.tokens'
  'vocab: tokens must be boxed list' assert 32 = 3!:0 raw_tokens
  ranks =. 0 $ a:
  scores =. 0 $ 0
  add_sp =. 0
  if. model -: 'llama' do.
    NB. Keep SPM pieces as raw UTF-8 byte strings (J u: rejects some astral tokens).
    tokens =. raw_tokens
    scores =. , load gguf_meta_default_jllamagguf_ 'tokenizer.ggml.scores' ; (0 $ 0)
    'vocab: llama SPM needs scores' assert (# scores) = # tokens
    bos =. {. load gguf_meta_default_jllamagguf_ 'tokenizer.ggml.bos_token_id' ; 1
    eos =. {. load gguf_meta_default_jllamagguf_ 'tokenizer.ggml.eos_token_id' ; 2
    unk =. {. load gguf_meta_default_jllamagguf_ 'tokenizer.ggml.unknown_token_id' ; 0
    NB. SPM defaults match llama.cpp: add_bos=1, space prefix on
    add_bos =. {. load gguf_meta_default_jllamagguf_ 'tokenizer.ggml.add_bos_token' ; 1
    add_eos =. {. load gguf_meta_default_jllamagguf_ 'tokenizer.ggml.add_eos_token' ; 0
    add_sp =. 1
    pre =. load gguf_meta_default_jllamagguf_ 'tokenizer.ggml.pre' ; 'default'
  else.
    tokens =. 0 $ a:
    for_t. raw_tokens do.
      tokens =. tokens , < utf8_decode > t
    end.
    merges =. load gguf_meta_default_jllamagguf_ 'tokenizer.ggml.merges' ; (0 $ a:)
    for_m. merges do.
      s =. utf8_decode > m
      sp =. s i. ' '
      'vocab: bad merge (no space)' assert sp < # s
      left =. sp {. s
      right =. (sp + 1) }. s
      ranks =. ranks , < (<left) , (<right)
    end.
    bos =. {. load gguf_meta_default_jllamagguf_ 'tokenizer.ggml.bos_token_id' ; 1
    eos =. {. load gguf_meta_default_jllamagguf_ 'tokenizer.ggml.eos_token_id' ; 2
    unk =. {. load gguf_meta_default_jllamagguf_ 'tokenizer.ggml.unknown_token_id' ; 0
    pre =. load gguf_meta_default_jllamagguf_ 'tokenizer.ggml.pre' ; 'byte'
    NB. Llama-3 BPE (pre=llama-bpe): llama.cpp adds BOS by default when meta omits add_bos_token.
    NB. Fixtures set add_bos_token=0 explicitly. Default 0 for other gpt2/bpe.
    def_bos =. pre -: 'llama-bpe'
    add_bos =. {. load gguf_meta_default_jllamagguf_ 'tokenizer.ggml.add_bos_token' ; def_bos
    add_eos =. {. load gguf_meta_default_jllamagguf_ 'tokenizer.ggml.add_eos_token' ; 0
  end.
  pack =. (<model) , (<tokens) , (<ranks) , (<bos) , (<eos) , (<unk) , (<add_bos) , (<add_eos) , (<pre) , (<scores) , (<add_sp)
  <"_ pack
)

vocab_from_gguf =: 3 : 0
  vocab_from_load gguf_load_jllamagguf_ y
)

vocab_open =: 3 : '> y'

vocab_tokens =: 3 : 0
  'model tokens ranks bos eos unk add_bos add_eos pre scores add_sp' =. vocab_open y
  tokens
)

vocab_bos =: 3 : 0
  'model tokens ranks bos eos unk add_bos add_eos pre scores add_sp' =. vocab_open y
  bos
)

vocab_eos =: 3 : 0
  'model tokens ranks bos eos unk add_bos add_eos pre scores add_sp' =. vocab_open y
  eos
)

vocab_unk =: 3 : 0
  'model tokens ranks bos eos unk add_bos add_eos pre scores add_sp' =. vocab_open y
  unk
)

vocab_token =: 4 : 0
  tokens =. vocab_tokens x
  id =. y
  'vocab_token: bad id' assert (id >: 0) *. id < # tokens
  > id { tokens
)

NB. ---------------------------------------------------------------
NB. SPM (llama) encode / decode — mirrors llama.cpp llm_tokenizer_spm
NB. ---------------------------------------------------------------

NB. hex digit char -> 0..15
hex_val =: 3 : 0
  c =. {. a. i. y
  if. (c >: 48) *. c <: 57 do. c - 48 return. end.
  if. (c >: 65) *. c <: 70 do. c - 55 return. end.
  if. (c >: 97) *. c <: 102 do. c - 87 return. end.
  _1
)

NB. piece -> byte 0..255 or _1 if not <0xNN>
spm_byte_piece =: 3 : 0
  t =. y
  if. 7 ~: # t do. _1 return. end.
  if. -. ('<0x' -: 3 {. t) *. '>' = {: t do. _1 return. end.
  h1 =. hex_val 3 { t
  h2 =. hex_val 4 { t
  if. (h1 < 0) +. h2 < 0 do. _1 return. end.
  h2 + 16 * h1
)

NB. tokens byte_to_token byte -> id
spm_byte_to_token =: 4 : 0
  tokens =. x
  b =. y
  hex =. '0123456789ABCDEF'
  h =. (hex {~ <. b % 16) , hex {~ 15 (17 b.) b
  piece =. '<0x' , h , '>'
  id =. tokens token_to_id piece
  if. id < # tokens do. id return. end.
  NB. fallback: single-byte char token
  id =. tokens token_to_id , b { a.
  if. id < # tokens do. id return. end.
  _1
)

NB. UTF-8 byte string: replace ASCII space (0x20) with U+2581 bytes E2 96 81
SPM_WS_BYTES =: (16be2 16b96 16b81) { a.

spm_escape_ws =: 3 : 0
  b =. , y
  if. 0 = # b do. '' return. end.
  out =. ''
  for_c. b do.
    if. ' ' = c do. out =. out , SPM_WS_BYTES else. out =. out , c end.
  end.
  out
)

NB. Replace U+2581 UTF-8 sequence with ASCII space (byte-level)
spm_unescape_ws =: 3 : 0
  b =. , y
  if. 0 = # b do. '' return. end.
  out =. ''
  i =. 0
  n =. # b
  while. i < n do.
    hit =. 0
    if. (i + 2) < n do.
      if. ((16be2 { a.) = i { b) *. ((16b96 { a.) = (i + 1) { b) *. ((16b81 { a.) = (i + 2) { b) do.
        hit =. 1
      end.
    end.
    if. hit do.
      out =. out , ' '
      i =. i + 3
    else.
      out =. out , i { b
      i =. i + 1
    end.
  end.
  out
)

NB. Split UTF-8 byte string into list of single-character UTF-8 pieces (boxed)
spm_chars =: 3 : 0
  b =. , y
  r =. 0 $ a:
  i =. 0
  n =. # b
  while. i < n do.
    c =. a. i. i { b
    if. c < 128 do. ln =. 1
    elseif. c < 224 do. ln =. 2
    elseif. c < 240 do. ln =. 3
    elseif. do. ln =. 4
    end.
    'vocab: truncated utf8 in spm_chars' assert (i + ln) <: n
    r =. r , < ln {. i }. b
    i =. i + ln
  end.
  r
)

NB. try_add helper state held in SPM_* locals of caller via pass-in boxes is hard;
NB. keep merge self-contained below.

NB. SPM merge on escaped unicode text -> list of token ids
NB. x = (<tokens),(<scores)
NB. Mirrors llama.cpp llm_tokenizer_spm_session (priority merges by score).
spm_merge_ids =: 4 : 0
  'tokens scores' =. x
  text =. y
  chars =. spm_chars text
  n =. # chars
  if. 0 = n do. 0 $ 0 return. end.
  sym_t =. chars
  sym_prev =. <: i. n
  sym_next =. 1 + i. n
  if. n do. sym_next =. (_1) (n - 1) } sym_next end.
  rev =. 0 $ a:
  q =. 0 $ a:

  NB. seed bigrams
  i =. 1
  while. i < n do.
    L =. i - 1
    R =. i
    t =. (> L { sym_t) , > R { sym_t
    id =. tokens token_to_id t
    if. id < # tokens do.
      q =. q , < ((id { scores) ; L ; R ; (# t))
      rev =. rev , < (t ; L ; R)
    end.
    i =. i + 1
  end.

  while. # q do.
    bi =. 0
    j =. 1
    while. j < # q do.
      'sc L R sz' =. > j { q
      'bsc bL bR bsz' =. > bi { q
      if. (sc > bsc) +. (sc = bsc) *. (L < bL) do. bi =. j end.
      j =. j + 1
    end.
    'sc L R sz' =. > bi { q
    q =. (bi {. q) , (bi + 1) }. q
    lt =. > L { sym_t
    rt =. > R { sym_t
    if. (0 = # lt) +. 0 = # rt do. continue. end.
    if. sz ~: (# lt) + # rt do. continue. end.
    sym_t =. (< lt , rt) L } sym_t
    sym_t =. (<'') R } sym_t
    nx =. R { sym_next
    sym_next =. nx L } sym_next
    if. nx >: 0 do. sym_prev =. L nx } sym_prev end.
    NB. try_add prev,L
    Lp =. L { sym_prev
    if. (Lp >: 0) *. (0 ~: # > Lp { sym_t) *. 0 ~: # > L { sym_t do.
      t =. (> Lp { sym_t) , > L { sym_t
      id =. tokens token_to_id t
      if. id < # tokens do.
        q =. q , < ((id { scores) ; Lp ; L ; (# t))
        rev =. rev , < (t ; Lp ; L)
      end.
    end.
    NB. try_add L,nx
    if. (nx >: 0) *. (0 ~: # > L { sym_t) *. 0 ~: # > nx { sym_t do.
      t =. (> L { sym_t) , > nx { sym_t
      id =. tokens token_to_id t
      if. id < # tokens do.
        q =. q , < ((id { scores) ; L ; nx ; (# t))
        rev =. rev , < (t ; L ; nx)
      end.
    end.
  end.

  NB. resegment with explicit stack (no nested verbs)
  out =. 0 $ 0
  stack =. 0 $ 0
  i =. 0
  while. i ~: _1 do.
    if. # > i { sym_t do. stack =. stack , i end.
    i =. i { sym_next
  end.
  NB. process left-to-right: stack currently first..last; reverse for LIFO of reverse order?
  NB. We pushed chain order; resegment each root in order using a work stack DFS.
  roots =. stack
  ri =. 0
  while. ri < # roots do.
    work =. , ri { roots
    while. # work do.
      idx =. {: work
      work =. }: work
      t =. > idx { sym_t
      id =. tokens token_to_id t
      if. id < # tokens do.
        out =. out , id
      else.
        found =. 0
        pL =. _1
        pR =. _1
        k =. <: # rev
        while. k >: 0 do.
          'rt a b' =. > k { rev
          if. t -: rt do.
            found =. 1
            pL =. a
            pR =. b
            break.
          end.
          k =. k - 1
        end.
        if. found do.
          NB. push right then left so left is processed first
          work =. work , pR , pL
        else.
          raw =. utf8_bytes t
          for_b. a. i. raw do.
            bid =. tokens spm_byte_to_token b
            'vocab: SPM byte fallback failed' assert bid >: 0
            out =. out , bid
          end.
        end.
      end.
    end.
    ri =. ri + 1
  end.
  out
)

encode_spm =: 4 : 0
  v =. x
  text =. y
  'model tokens ranks bos eos unk add_bos add_eos pre scores add_sp' =. vocab_open v
  ids =. 0 $ 0
  if. add_bos do. ids =. ids , bos end.
  if. # text do.
    t =. utf8_bytes text
    if. add_sp do. t =. ' ' , t end.
    t =. spm_escape_ws t
    ids =. ids , ((<tokens) , (<scores)) spm_merge_ids t
  end.
  if. add_eos do. ids =. ids , eos end.
  ids
)

decode_spm =: 4 : 0
  v =. x
  ids =. , y
  'model tokens ranks bos eos unk add_bos add_eos pre scores add_sp' =. vocab_open v
  pieces =. ''
  for_id. ids do.
    if. (id = bos) +. id = eos do. continue. end.
    if. (id < 0) +. id >: # tokens do. continue. end.
    t =. > id { tokens
    b =. spm_byte_piece t
    if. b >: 0 do.
      pieces =. pieces , b { a.
    else.
      pieces =. pieces , spm_unescape_ws t
    end.
  end.
  NB. Prefer unicode text; fall back to raw bytes if J rejects a codepoint.
  try. utf8_decode pieces catch. pieces end.
)

NB. ---------------------------------------------------------------
NB. encode / decode dispatch
NB. ---------------------------------------------------------------

encode_bpe =: 4 : 0
  v =. x
  text =. y
  'model tokens ranks bos eos unk add_bos add_eos pre scores add_sp' =. vocab_open v
  ids =. 0 $ 0
  if. add_bos do. ids =. ids , bos end.
  if. # text do.
    raw =. utf8_bytes text
    chars =. bytes_to_bpe_chars raw
    word =. 0 $ a:
    for_c. chars do. word =. word , < , c end.
    word =. ranks bpe_merge word
    for_s. word do.
      t =. > s
      id =. tokens token_to_id t
      if. id >: # tokens do. id =. unk end.
      ids =. ids , id
    end.
  end.
  if. add_eos do. ids =. ids , eos end.
  ids
)

decode_bpe =: 4 : 0
  v =. x
  ids =. , y
  'model tokens ranks bos eos unk add_bos add_eos pre scores add_sp' =. vocab_open v
  pieces =. ''
  for_id. ids do.
    if. (id = bos) +. id = eos do. continue. end.
    if. (id < 0) +. id >: # tokens do.
      t =. > unk { tokens
    else.
      t =. > id { tokens
    end.
    pieces =. pieces , t
  end.
  b =. bpe_chars_to_bytes pieces
  bytes_to_text b
)

NB. vocab encode text
encode =: 4 : 0
  v =. x
  'model tokens ranks bos eos unk add_bos add_eos pre scores add_sp' =. vocab_open v
  if. model -: 'llama' do.
    v encode_spm y
  else.
    v encode_bpe y
  end.
)

NB. vocab decode ids
decode =: 4 : 0
  v =. x
  'model tokens ranks bos eos unk add_bos add_eos pre scores add_sp' =. vocab_open v
  if. model -: 'llama' do.
    v decode_spm y
  else.
    v decode_bpe y
  end.
)

cocurrent 'base'
