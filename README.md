# jllama

Llama-style dense decoder inference in the **J** programming language (Jsoftware J9.8).

jllama loads a Llama-architecture GGUF (F16/F32), tokenizes a prompt, runs prefill + KV-cache decode, and prints generated text. The transformer is implemented as ordinary J (`core/`, `io/`, `cli/`) — not a port of llama.cpp.

## Requirements

- **J 9.8** with jconsole at a full path (on macOS typically `/Applications/j9.8/bin/jconsole`). Do not use bare `jconsole` if that resolves to the Java tool.
- A **Llama** GGUF (`general.architecture=llama`) in **F16** or **F32** (quantized GGUFs are not supported yet).
- Enough RAM for F16→**float64** weights in J (roughly 4× the F16 file size for the weight tensors, plus activations/KV).

## Setup

From this directory, publish the engine into J user temp so scripts can
`load jpath '~temp/jllama/...'`:

```sh
./publish_jllama
# -> ~/j9.8-user/temp/jllama/{sysutils,jllama_cli,core,io,cli}/...
```

Re-run after you edit any runtime `.ijs` file.

## Run

`jllama_cli.ijs` is a standalone shebang script. Place models under `models/` (gitignored) or pass any path to `-m`.

### TinyStories (~15M, ~47 MB F16) — quick smoke

```sh
# once:
export PATH="$HOME/.local/bin:$PATH"   # after: uv tool install huggingface_hub
hf download shibatch/stories-converted stories15M.F16.gguf --local-dir models

./publish_jllama
./jllama_cli.ijs -m models/stories15M.F16.gguf -p "Once upon a time" -n 32
```

### Llama 3.2 1B Instruct (F16)

```sh
# once (example source; any Llama-arch F16 GGUF works):
export PATH="$HOME/.local/bin:$PATH"
hf download bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-f16.gguf --local-dir models

./publish_jllama
./jllama_cli.ijs -m models/Llama-3.2-1B-Instruct-f16.gguf -p "Once upon a time" -n 32
```

Notes for the 1B model:

- First run is slow in pure J f64 (load + prefill); keep `-n` modest at first.
- No chat template — prompts are raw text (you may wrap Instruct prompts in Llama-3 headers yourself).
- BOS is prepended for Llama-3 BPE (`pre=llama-bpe`).

### Help

```sh
./jllama_cli.ijs --help
./jllama_cli.ijs --version
```

Without the shebang, call jconsole explicitly:

```sh
/Applications/j9.8/bin/jconsole jllama_cli.ijs -m models/stories15M.F16.gguf -p "Once upon a time" -n 32
```

### Flags

| Flag | Meaning |
|------|---------|
| `-m, --model PATH` | GGUF model (**required**) |
| `-p, --prompt TEXT` | prompt string |
| `-f, --file PATH` | read prompt from file |
| `-n, --n-predict N` | new tokens to generate (default 16) |
| `--temp F` | temperature; `<=0` is greedy (default 0) |
| `--top-k K` | top-k; `<=0` off |
| `--top-p P` | nucleus; `>=1` off |
| `--seed S` | RNG seed |
| `--eos ID` | EOS token id (default from model vocab) |
| `--no-stop` | do not stop when EOS is sampled |
| `--tokens` | also print token ids |
| `-h, --help` | help |
| `--version` | version and exit |

Use either `-p` or `-f` for the prompt.

## What is supported

| Supported | Not in this tree |
|-----------|------------------|
| `general.architecture=llama` dense models | Other GGUF arches (Qwen, Gemma, …) |
| F16 / F32 weights | Quant (Q4/Q5/Q8, …) |
| MHA and GQA (`n_head_kv`) | GPU / Metal backends |
| RMSNorm, RoPE (NORMAL), SwiGLU | Chat templates / GBNF |
| Greedy and temp / top-k / top-p sampling | Training, server mode |

On a 32 GB machine, **~1B F16** is a practical upper lab size in pure J f64; **7B+ F16** is not realistic without quant or an external backend.

## Runtime layout

```text
jllama_cli.ijs     standalone CLI (shebang → jconsole)
publish_jllama     copy engine → jpath '~temp/jllama'
sysutils.ijs       ROOT, setroot, jrequire*, VERSION (locale jllamasys)
cli/cli.ijs        argv parse + generate driver (locale jllamacli)
core/              tensor, rope, attention, block, sample, model
io/                GGUF loader, vocab (BPE / Llama SPM)
models/            your GGUFs (not shipped; see models/.gitignore)
```

After `./publish_jllama`, loads resolve under:

```text
~/j9.8-user/temp/jllama/    NB. jpath '~temp/jllama'
```

## License

TBD.
