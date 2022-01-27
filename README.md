# Accelerating Backward Aggregation in GCN Training with Execution Path Preparing on GPUs

## 1. Getting Started Instructions.

+ **Clone this project**
```shell
git clone https://github.com/Catriminal/GNNAdvisor.git
```

+ **Hardware**: 
> + `CPU x86_64` with host memory >= 32GB. (Tested on Intel Xeon E5-2680 v4 (14-core 28-thread)  CPU  with 256 GB host memory).
> + `NVIDIA GPU (arch>=sm_60)` with device memory >= 16GB. Tested on NVIDIA Tesla P100(`sm_60`). 

+ **OS & Compiler**: 
> + `Ubuntu 16.04+`
> + `gcc >= 7.5`
> + `cmake >= 3.14`
> + `CUDA >= 10.1` and `nvcc >= 10.0`

+ **Important Files/Directories**

> + `EPPGCN/`: the directory for our system and Python benchmark. 
> > + `GCNConv/`: the C++/CUDA source code (`EPPGCN_kernel.cu`, `cuCompactor.cu`) for GCN sparse computation kernel and graph compaction, python binding of kernels (`EPPGCN.cpp`) and python `setup.py` installation script.
> > + `gcn_conv.py`: the Python script for defining the GCN convolution at high-level.
> > + `param.py`: the Python script for defining the input-level properties and different rules for handling this properties to generate performance-related configuration.
> > + `dataset.py`: the Python loader for datasets from either plain `.txt` edgeList files or binary `.npy` file.


### **Step-1: Environment Setup** 

#### 1) Install Pytorch environment.
+ Install **`conda`** on system **[Tutorial](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)**.
+ Create a **`conda`** environment: 
```shell
conda create -n env_name python=3.6
```
+ Install **`Pytorch`**: 
```shell
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
```
or using `pip` [**Note that make sure the `pip` you use is the `pip` from current conda environment. You can check this by `which pip`**]
```shell
pip install torch==1.7.0+cu101 torchvision==0.8.0+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install tqdm
pip install scipy
```
+ Install [**`Pytorch-Geometric (PyG)`**](https://github.com/rusty1s/pytorch_geometric).
```shell
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip install torch-geometric
```

+ Install our system Pytorch Binding.
> + Go to `EPPGCN/GCNConv`, then `python setup.py install` to install the our system modules.

### **Step-2: Download the graph datasets.**
+ The graph data files we used can all be found in the references of the paper.
+ The format of the graph data file should be SNAP, using '\t' as a separator for each row of vertex ids.
+ We use mask to represent the training set. The format of the mask file should be a text file consisting of 0s and 1s, with the number of lines equal to the number of vertices of the graph.
+ Note the modification of the data storage path in `bench_EPPGCN.py`.
+ Note that node inital embeeding is not included, and we generate an all 1s embeeding matrix according to users `input dimension` parameter at the runtime for just performance evaluation.

## 3. Detailed Instructions.

+ **GNN Model Setting**.
> + **GCN (multi-layer with 16 hidden dimension)**
+ **Datasets**.

> `cora, citeseer, pubmed, dblp, youtube, amazon, corafull, catalog, twitter, google`

+ **Running baseline GNNAdvisor**.
> + Checkout git branch to initial.
>
>   ```shell
>   git checkout initial
>   ```
>
> + Go to `GNNAdvisor/GNNConv` directory.
>
> + Install GNNAdvisor Pytorch Binding.
>
>   ```shell
>   python setup.py install
>   ```
>
> + Go to `GNNAdvisor/` directory.
>
> + `./0_bench_GNNA_GCN.py` to run baseline GNNAdvisor's 2-layer GCN model and report 100 epoch runtime for all evaluated datasets.

+ **Running our system**.
> + Checkout git branch to master.
>
>   ```shell
>   git checkout master
>   ```
>
> + Go to `EPPGCN/GCNConv` directory.
>
> + Install our system Pytorch Binding.
>
>   ```shell
>   python setup.py install
>   ```
>
> + Go to `EPPGCN/` directory. 
>
> + `./bench_EPPGCN.py` to run our system multi-layer GCN model and report 100 epoch runtime for all evaluated datasets in different train ratios and numbers of layers.
>
> + Set parameters `--groupsize_model`  to choose different ways to decide `groupsize` in backward process. There are three options for this parameter: `regression_equation`, `SAGPG` and `fixed_value`. `regression_equation` uses a linear regression equation, `SAGPG` uses a trained neural network and `fixed_value` uses a fixed constant of 32.
>
> +  Stand alone running `GNNA_main.py` with specified parameters.
> > + `--dataset`: the name of the dataset.
> > + `--dim`: the size of input embedding dimension, default: 96.
> > + `--hidden`: the size of hidden dimension, default: 16.
> > + `--classes`: the number of output classes, default: 22.
> > + `--layers`: the number of layers, default: 2.
> > + `--train_ratio`: the ratio of training set, default: 0.1
> > + `--num_epoches`: the number of epoches for training, default: 100.


# Reference
+ [**Pytorch Geometric**](https://github.com/rusty1s/pytorch_geometric) <br>
  Fey, Matthias, and Jan Eric Lenssen. 
  **Fast graph representation learning with PyTorch Geometric.** 
  *The International Conference on Learning Representations (ICLR), 2019.*
+ [**GNNAdvisor**](https://github.com/YukeWang96/OSDI21_AE)<br>
  Yuke Wang, Boyuan Feng, et al. 
  **GNNAdvisor: An Efficient Runtime System for GNN Acceleration on GPUs**. *USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2021*