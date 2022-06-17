# Argument Mining Project

## Things to note:
- Code only tested on Pop_OS 21.10 (Ubuntu)
- Linux kernel 5.17.5-76051705-generic
- Only tested with Python 3.10

## Dataset link:

## Code references:

## Prerequisites befores running:
- Install requirements in requirements.txt
- run setup.py

## Setting up PyCharm on Linux:
- Download Pycharm
- Add to ~/.bashrc
  - `export PATH=$PATH:/path_to_pycharm/PyCharm/bin/`
- run `source ~/.bashrc`
- run Pycharm in Project directory using terminal `./pycharm.sh`

## Getting GoogleNews-vectors
1. `python -m gensim.downloader -d word2vec-google-news-300`
2. Move word2vec-google-news-300.gz to Project/ directory 
3. gzip -d ./word2vec-google-news-300.gz
4. https://code.google.com/archive/p/word2vec/
