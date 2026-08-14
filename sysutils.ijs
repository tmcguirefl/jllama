NB. jllama system utilities
NB. Locale: jllamasys
NB.
NB. Shared bootstrap used by jllama_cli.ijs and jllama_dev.ijs.
NB.   setroot marker   - set ROOT from a loaded script path (4!:3)
NB.   jrequire relpath - load ROOT , relpath  (project tree: tests/fixtures)
NB.   jrequire_temp relpath - load jpath '~temp/jllama/' , relpath
NB.   jtemp relpath    - full path under published ~temp/jllama
NB.   ROOT             - project (or entry) root with trailing /
NB.   VERSION MILESTONE
NB.
NB. Publish engine scripts with:  ./publish_jllama
NB. Then:  load jpath '~temp/jllama/sysutils.ijs'

cocurrent 'jllamasys'

VERSION =: '0.13.3'
MILESTONE =: 'M13'

NB. Placeholder until setroot runs.
ROOT =: (1!:43 '') , '/'

NB. y = basename (or unique substring) of the entry script already in 4!:3,
NB. e.g. 'jllama_cli.ijs' or 'jllama_dev.ijs'.
NB. Sets ROOT to that script's directory (trailing /) and returns ROOT.
setroot =: 3 : 0
  marker =. y
  'setroot: marker required' assert 0 < # marker
  scripts =. 4!:3 ''
  hit =. scripts #~ +./@(marker&E.)@> scripts
  if. # hit do.
    p =. > {: hit
    ROOT =: ((p i: '/') {. p) , '/'
  else.
    ROOT =: (1!:43 '') , '/'
  end.
  ROOT
)

NB. Full path under published tree: jtemp 'core/tensor.ijs'
jtemp =: 3 : 0
  (jpath '~temp/jllama/') , y
)

NB. Load a project script relative to ROOT (fixtures, tests, labs).
jrequire =: 3 : 0
  path =. ROOT , y
  if. -. fexist path do.
    echo 'jllama: missing ' , path
    'missing script' assert 0
  end.
  load path
)

NB. Load a published engine script from ~temp/jllama.
jrequire_temp =: 3 : 0
  path =. jtemp y
  if. -. fexist path do.
    echo 'jllama: missing published ' , path
    echo 'jllama: run ./publish_jllama from the repo'
    'missing published script' assert 0
  end.
  load path
)

NB. Convenience: return ROOT.
root =: 3 : 'ROOT'

cocurrent 'base'
