# cli/

Shell front-end for jllama (M8).

| File | Role |
|------|------|
| `cli.ijs` | locale `jllamacli` — parse args, load model, generate, print |
| `../jllama_cli.ijs` | standalone shebang entry (load + `main` + exit) |

## Usage

```sh
./jllama_cli.ijs -m models/stories15M.F16.gguf -p "Once upon a time" -n 32
./jllama_cli.ijs -m models/Llama-3.2-1B-Instruct-f16.gguf -p "Hello" -n 16
./jllama_cli.ijs --help
```

Or via jconsole:

```sh
/Applications/j9.8/bin/jconsole jllama_cli.ijs -m MODEL.gguf -p hi -n 8
```

## Locale `jllamacli`

| Verb | Role |
|------|------|
| `parse_args` | boxed argv → opts list |
| `run_opts` | opts → exit code (prints text) |
| `run` | argv → exit code (try/catch) |
| `main` | `cli_argv` + `2!:55` |
| `cli_argv` | strip jconsole + `*.ijs` from `ARGV_z_` |

Sampling flags map to M7 cfg: `temp ; top_k ; top_p ; seed ; eos_id ; stop_on_eos`.
