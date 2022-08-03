# Iolaus

Iolaus implements an optimization to cross-validation by using backtracking on a distributed ML training framework.

## Design Overview

Iolaus attempts to improve upon traditional grid-search, which is commonly used in conjunction with cross-validation, in order to obtain more appropriately-tuned values that account for the variance in the dataset. The main idea of Iolaus is that by running multiple cross-validation jobs currently in a distributed setting, it is possible to optimize what parameters are run on each job and therefore eliminate combinations of hyperparameters through certain heuristics, improving overall performance.

Similar work in parallelizing hyperparameter optimization has been done in [this paper](https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/). 

## Environment Setup

To install the virtual environment, run the following commands from the project's root directory:
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
