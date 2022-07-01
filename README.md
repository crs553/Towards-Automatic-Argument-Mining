# Argument Mining Project

## Things to note:
- Code only tested on Pop_OS 21.10 (Ubuntu)
- Linux kernel 5.17.5-76051705-generic
- Only tested with Python 3.10

## Dataset link:
- [Student Essay Corpus v2](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422)

## Code references:
- [ArgumentBert](https://github.com/negedng/argument_BERT/)
- [MilzParser](https://github.com/Milzi/arguEParser)

## Prerequisites befores running:
- Install requirements in requirements.txt
- run setup.py
- Run Spacy download command (at bottom of readme)

## Setting up PyCharm on Linux:
- Download Pycharm
- Add to ~/.bashrc
  - `export PATH=$PATH:/path_to_pycharm/PyCharm/bin/`
- run `source ~/.bashrc`
- run Pycharm in Project directory using terminal `./pycharm.sh`


## Spacy Download
1. python -m spacy download en_core_web_lg
