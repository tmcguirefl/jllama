NB. jllama CLI (M8)
NB.
NB. Locale: jllamacli
NB.
NB. Public:
NB.   usage ''
NB.   parse_args boxed_argv_list
NB.   run_opts open_opts_list
NB.   run boxed_argv_list   -> exit code (0 ok, 2 usage/args, 1 error)
NB.   main ''               -> 2!:55 with code (from ARGV_z_)
NB.   cli_argv ''           -> args after jconsole + scripts
NB.
NB. opts open list (12 items via ; ):
NB.   model ; prompt ; n_new ; temp ; top_k ; top_p ; seed ;
NB.   eos_id ; stop_on_eos ; show_tokens ; do_help ; do_version
NB. eos_id = _2 means "use vocab_eos after load"

cocurrent 'jllamacli'

model_from_gguf =: model_from_gguf_jllamagguf_
vocab_from_gguf =: vocab_from_gguf_jllamavocab_
encode =: encode_jllamavocab_
decode =: decode_jllamavocab_
vocab_eos =: vocab_eos_jllamavocab_
generate_sample =: generate_sample_jllamamodel_

NB. ---------------------------------------------------------------
NB. Defaults / usage
NB. ---------------------------------------------------------------

NB. eos _2 = auto from vocab
default_opts =: 3 : 0
  '' ; '' ; 16 ; 0 ; 0 ; 1 ; 0 ; _2 ; 1 ; 0 ; 0 ; 0
)

USAGE =: 0 : 0
jllama_cli - Llama-style generate in J

Usage:
  jllama_cli -m MODEL.gguf -p PROMPT [options]
  jllama_cli -m MODEL.gguf -f PROMPTFILE [options]

Options:
  -m, --model PATH     GGUF model (required)
  -p, --prompt TEXT    prompt string
  -f, --file PATH      read prompt from file
  -n, --n-predict N    new tokens to generate (default 16)
      --temp F         temperature; <=0 greedy (default 0)
      --top-k K        top-k; <=0 off (default 0)
      --top-p P        top-p nucleus; >=1 off (default 1)
      --seed S         RNG seed (default 0)
      --eos ID         EOS token id (default: model vocab eos)
      --no-stop        do not stop when EOS is sampled
      --tokens         also print token ids (space-separated)
  -h, --help           this help
      --version        print version and exit

Examples:
  jllama_cli -m test/fixtures/tiny_parity_f16.gguf -p ab -n 3
  jllama_cli -m model.gguf -p "Hello" -n 32 --temp 0.8 --top-k 40 --top-p 0.95
)

usage =: 3 : 0
  smoutput USAGE
  i. 0 0
)

NB. ---------------------------------------------------------------
NB. ARGV helpers
NB. ---------------------------------------------------------------

NB. Drop jconsole binary and any leading *.ijs scripts; rest are flags.
cli_argv =: 3 : 0
  a =. ARGV_z_
  i =. 1
  n =. # a
  while. i < n do.
    s =. > i { a
    if. (4 < # s) *. '.ijs' -: _4 {. s do.
      i =. i + 1
    else.
      break.
    end.
  end.
  i }. a
)

NB. boxed string -> number (scalar); empty fails
read_num =: 3 : 0
  s =. , > y
  'cli: empty number' assert # s
  v =. ". s
  'cli: bad number ' , s assert 0 = #$ v
  v
)

NB. x = flag name (for errors)
NB. y = (<args) , (<i)   args = boxed argv list, i = index of flag
NB. returns (<value_box) , (<next_i)  where value_box is ARGV element box
need_val =: 4 : 0
  flag =. x
  args =. > 0 { y
  i =. > 1 { y
  n =. # args
  'cli: missing value for ' , flag assert (i + 1) < n
  ((i + 1) { args) , (<i + 2)
)

NB. ---------------------------------------------------------------
NB. parse_args
NB. y = list of boxed char vectors (CLI flags only)
NB. returns open opts list (see header)
NB. ---------------------------------------------------------------
parse_args =: 3 : 0
  args =. y
  NB. normalize to a list of boxed strings (scalar box -> 1-list)
  if. 32 = 3!:0 args do.
    if. 0 = #$ args do. args =. , < > args
    elseif. 0 = # args do. args =. 0 $ a:
    end.
  else.
    if. 0 = # , args do. args =. 0 $ a:
    else. args =. , < , args
    end.
  end.
  o =. default_opts ''
  'model prompt n_new temp top_k top_p seed eos_id stop show_tok do_help do_ver' =. o
  i =. 0
  n =. # args
  while. i < n do.
    a =. > i { args
    select. a
    case. '-h'; '--help' do.
      do_help =. 1
      i =. i + 1
    case. '--version' do.
      do_ver =. 1
      i =. i + 1
    case. '--tokens' do.
      show_tok =. 1
      i =. i + 1
    case. '--no-stop' do.
      stop =. 0
      i =. i + 1
    case. '-m'; '--model' do.
      NB. multi-assign opens one box level: vb is the char vector
      'vb i' =. a need_val (<args) , (<i)
      model =. vb
    case. '-p'; '--prompt' do.
      'vb i' =. a need_val (<args) , (<i)
      prompt =. vb
    case. '-f'; '--file' do.
      'vb i' =. a need_val (<args) , (<i)
      path =. vb
      'cli: prompt file missing: ' , path assert fexist path
      prompt =. 1!:1 < path
      if. (# prompt) *. (10{a.) = {: prompt do. prompt =. }: prompt end.
      if. (# prompt) *. (13{a.) = {: prompt do. prompt =. }: prompt end.
    case. '-n'; '--n-predict' do.
      'vb i' =. a need_val (<args) , (<i)
      n_new =. <. read_num < vb
    case. '--temp' do.
      'vb i' =. a need_val (<args) , (<i)
      temp =. read_num < vb
    case. '--top-k' do.
      'vb i' =. a need_val (<args) , (<i)
      top_k =. <. read_num < vb
    case. '--top-p' do.
      'vb i' =. a need_val (<args) , (<i)
      top_p =. read_num < vb
    case. '--seed' do.
      'vb i' =. a need_val (<args) , (<i)
      seed =. <. read_num < vb
    case. '--eos' do.
      'vb i' =. a need_val (<args) , (<i)
      eos_id =. <. read_num < vb
    case. do.
      'cli: unknown argument: ' , a assert 0
    end.
  end.
  model ; prompt ; n_new ; temp ; top_k ; top_p ; seed ; eos_id ; stop ; show_tok ; do_help ; do_ver
)

NB. Format int list as space-separated decimal text
fmt_ids =: 3 : 0
  ids =. , y
  if. 0 = # ids do. '' return. end.
  out =. ": 0 { ids
  for_k. 1 + i. <: # ids do.
    out =. out , ' ' , ": k { ids
  end.
  out
)

NB. ---------------------------------------------------------------
NB. run_opts / run
NB. ---------------------------------------------------------------

NB. y = open opts from parse_args
NB. returns exit code 0/1/2
run_opts =: 3 : 0
  'model prompt n_new temp top_k top_p seed eos_id stop show_tok do_help do_ver' =. y
  if. do_help do.
    usage ''
    0 return.
  end.
  if. do_ver do.
    smoutput 'jllama_cli ' , VERSION_jllamasys_ , ' (' , MILESTONE_jllamasys_ , ')'
    0 return.
  end.
  if. 0 = # model do.
    smoutput 'jllama_cli: -m / --model is required'
    smoutput 'Try: jllama_cli --help'
    2 return.
  end.
  if. -. fexist model do.
    smoutput 'jllama_cli: model not found: ' , model
    1 return.
  end.
  if. n_new < 0 do.
    smoutput 'jllama_cli: -n must be >= 0'
    2 return.
  end.

  m =. model_from_gguf model
  v =. vocab_from_gguf model
  ids =. v encode prompt
  if. 0 = # ids do.
    smoutput 'jllama_cli: empty token ids after encode (empty prompt?)'
    1 return.
  end.

  if. eos_id = _2 do. eos_id =. vocab_eos v end.
  cfg =. temp ; top_k ; top_p ; seed ; eos_id ; stop
  out =. m generate_sample (<ids) , (<n_new) , (<cfg)
  text =. v decode out
  if. show_tok do.
    smoutput 'tokens: ' , fmt_ids out
  end.
  smoutput text
  0
)

NB. y = boxed argv (flags only)
run =: 3 : 0
  try.
    o =. parse_args y
    run_opts o
  catch.
    smoutput 'jllama_cli: ' , deb 13!:12 ''
    1
  end.
)

NB. Entry for jllama_cli.ijs — always process exit
main =: 3 : 0
  rc =. run cli_argv ''
  2!:55 rc
)
