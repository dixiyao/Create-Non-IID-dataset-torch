# Create Non IID dataset
## Introduction
This is a demo code about how to creating non iid datasets in federated learning, using Dirichelet distribution. It is a common way to use Dirichelet distribution process to construct non iid datasets with centralized dataset, e.g. Cifar10. Dir(α)，smaller α represents stronger non iid.

This code is used for generating non iid dataset in paper

> Dixi Yao*, Lingdong Wang*, Jiayu Xu, Liyao Xiang, Shuo Shao, Yingqi Chen, Yanjun Tong.   
> **Federated Model Search via Reinforcement Learning**  
> _International Conference on Distributed Computing Systems 2021_.

[Paper](https://ieeexplore.ieee.org/document/9546522)

## Requirements
numpy  
torch

## Usage
function `partition_data` in the file `noniid.py`, direcly appoints the location of data dir and generate a list of dataloaders, each data loader for each client in FL.
A sample usage
```python
X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data('./data',10)
```
## Further Improvement
Currently, the size of the datasets among each client is random or averaged. In the acutal Federated Learning Setting, sometimes the dataset size among clients are different. Further imporvement will be about how to appoint the size of dataset on each client. 
