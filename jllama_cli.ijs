#!/Applications/j9.8/bin/jconsole
NB. jllama_cli - standalone shell entry (M8)
NB.
NB. Prefer (after ./publish_jllama):
NB.   ./jllama_cli.ijs -m MODEL.gguf -p PROMPT ...
NB.
NB. Shebang runs jconsole on this file. Loads published modules via
NB.   jpath '~temp/jllama/...'
NB. so loads are static (top-level), not only inside explicit defs.
NB. No dependency on jllama_dev.ijs.

cocurrent 'base'

load jpath '~temp/jllama/sysutils.ijs'
setroot_jllamasys_ 'jllama_cli.ijs'

NB. Engine load order (tensor is pulled in by attention/block/sample/model).
load jpath '~temp/jllama/core/rope.ijs'
load jpath '~temp/jllama/core/attention.ijs'
load jpath '~temp/jllama/core/block.ijs'
load jpath '~temp/jllama/core/sample.ijs'
load jpath '~temp/jllama/core/model.ijs'
load jpath '~temp/jllama/io/gguf.ijs'
load jpath '~temp/jllama/io/vocab.ijs'
load jpath '~temp/jllama/cli/cli.ijs'

main_jllamacli_ ''
