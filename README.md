# SEComm

The official PyTorch implementation of **S**elf-**E**xpressive Graph Neural Network for Unsupervised
**Comm**unity Detection.

We have used the pytorch implementation of GRACE(https://github.com/CRIPAC-DIG/GRACE) for our self supervised GNN module.

## Prerequisites
Cora, CiteSeer, PubMed and Physics datasets are sourced from Pytorch Geometric. To install pytorch and pytorch geometric correctly - follow this tutorial(https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## Usage

Train and evaluate the model by executing
```
python train.py --dataset Cora 
```
The `--dataset` argument should be one of [Cora, CiteSeer, PubMed, Wiki, Physics].
<br>The `--pretrain` argument should be one of [T, T1, F].
T - means full training, T1-to skip the pretraining part(load already saved GNN model) and run the rest, F - to skip pretraining as well as self expressive layer training(ie. load the saved GNN model as well as node similarities list).
<br>The hyperparameters used are set in the file config.yaml.
