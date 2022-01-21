# Create Non IID dataset
The demo code of creating non iid dataset based on dirichelet distribution. Use Pytorch Framework

## Usage
function `partition_data` in the file `noniid.py`, direcly appoints the location of data dir and generate a list of dataloaders, each data loader for each client in FL.
A sample usage
```
X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data('./data',10)
```
