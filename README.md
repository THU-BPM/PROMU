# PROMU

The source code of paper "Prompt Me Up: Unleashing the Power of Alignments for Multimodal Entity and Relation Extraction"

## Installation

For training, a GPU is recommended to accelerate the training speed.

### PyTroch

The code is based on PyTorch 1.6+. You can find tutorials [here](https://pytorch.org/tutorials/).

### Dependencies

The dependencies are summarized in the file ```requirements.txt```. 

You can install these dependencies like this:

```
pip3 install -r requirements.txt
```

You also need to install [Horovod](https://github.com/horovod/horovod) and [Apex](https://github.com/NVIDIA/apex.git)
## Usage
* Run the full model on SemEval dataset with default hyperparameter settings<br>

```sh run.sh```<br>

## Data Download

* NewsCLIPpings: Automatic Generation of Out-of-Context Multimodal Media ([access](https://github.com/g-luo/news_clippings))<br>

