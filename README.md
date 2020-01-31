# SentRep

A place for developing sentence representations. The aim of this repository is to improve sentence representations
with self-supervised training of sentence representations. We start from sentence-BERT, a simplistic sentence encoding
scheme which fine-tunes BERT on SNLI and MultiNLI data.

# Usage

This repository utilizes PyTorch as well as the Huggingface transformers package for BERT training.

With CUDA available, run 

```python3 train.py```

to train the current model. Configuration parameters can be adjusted in config.py,
or passed as command-line arguments using ArgParse.

Similarly, run

```python3 eval.py```

to run SentEval evaluations on a variety of tasks, see https://github.com/facebookresearch/SentEval for more details.


