# Iolaus

## Introduction

Iolaus implements an optimization to cross-validation by using backtracking on a distributed ML training framework.

## Environment Setup

Install dev environment:
```bash
conda env create -f environment.yml
conda activate iolaus-dev
conda env list
```

If this is your first time running Iolaus, you may encounter this error:
```
Traceback (most recent call last):
  File "/Users/Jeremy/Desktop/iolaus/src/user_api_test.py", line 44, in <module>
    train_data = datasets.MNIST(
  File "/Users/Jeremy/opt/anaconda3/envs/iolaus-dev/lib/python3.10/site-packages/torchvision/datasets/mnist.py", line 102, in __init__
    raise RuntimeError("Dataset not found. You can use download=True to download it")
```

In that case, follow the instructions to modify the Torchvision source and rerun. If all goes well, then the MNIST dataset should download locally.

