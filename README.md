# Argument Mining Project

## 3rd Year Project University of York
- Project created by: Charles Stubbs
- Supervisor: [Tommy Yuan](https://www-users.cs.york.ac.uk/~tommy/)

## Abstract
This paper addresses the labour-intensive nature and expert annotator requirements of manual argument annotation by providing an initial sentence-level pipeline for argument identification and sentence relationship prediction. Before establishing our approach, we review current efforts to produce combined techniques and the available abstract pipelines. Next, we use the methods indicated in this review to produce standalone identification and relationship prediction methods. Finally, we apply these methods in a pipeline to produce a combined approach. The presented results demonstrate effective sentence identification but shortcomings in relationship detection. Overall, this paper suggests that combined approaches to argumentation are a promising area requiring further research.

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
