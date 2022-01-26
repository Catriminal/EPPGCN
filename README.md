# GNNAdvisor

## 1. Getting Started Instructions.
+ **Clone this project**
```shell
git clone https://github.com/Catriminal/GNNAdvisor.git
```

+ **Hardware**: 
> + `CPU x86_64` with host memory >= 32GB. (Tested on Intel Xeon Silver 4110 (8-core 16-thread)  CPU  with 64GB host memory).
> + `NVIDIA GPU (arch>=sm_60)` with devcie memory >= 16GB. Tested on NVIDIA Tesla P100(`sm_70`). 

+ **OS & Compiler**: 
> + `Ubuntu 16.04+`
> + `gcc >= 7.5`
> + `cmake >= 3.14`
> + `CUDA >= 10.1` and `nvcc >= 10.0`

+ **Important Files/Directories**

> + `rabbit_module/`: contains the source of rabbit reordering and python binding.
> + `GNNAdvisor/`: the directory for GNNAdvisor and Python benchmark. 
> > + `GNNConv/`: the C++/CUDA source code (`GNNAdvisor_kernel.cu`, `cuCompactor.cu`) for GCN sparse computation kernel and graph compaction, python binding of kernels (`GNNAdvisor.cpp`) and python `setup.py` installation script.
> > + `gnn_conv.py`: the Python script for defining the GCN convolution at high-level.
> > + `param.py`: the Python script for defining the input-level properties and different rules for handling this properties to generate performance-related configuration, such as `warpPerBlock`.
> > + `dataset.py`: the Python loader for datasets from either plain `.txt` edgeList files or binary `.npy` file.


### **Step-1: Environment Setup** 
#### 1) Install system packages for compiling rabbit reordering (root user required). 
+ **`libboost`**: `sudo apt-get install libboost-all-dev`.
+ **`tcmalloc`**: `sudo apt-get install libgoogle-perftools-dev`.
+ **`cmake`**: `sudo apt-get update && sudo apt-get -y install cmake protobuf-compiler`.


#### 2) Install Pytorch environment.
+ Install **`conda`** on system **[Toturial](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)**.
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
+ Install [**`Deep Graph Library (DGL)`**](https://github.com/dmlc/dgl).
```shell
conda install -c dglteam dgl-cuda10.1
pip install torch requests
```

+ Install [**`Pytorch-Geometric (PyG)`**](https://github.com/rusty1s/pytorch_geometric).
```shell
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip install torch-geometric
```

+ Install GNNAdvisor Pytorch Binding.
> + Go to `GNNAdvisor/GNNConv`, then `python setup.py install` to install the GNNAdvisor modules.
> + Go to `rabbit_module/src`, then `python setup.py install` to install the rabbit reordering modules.

### **Step-2: Download the graph datasets.**
+ The graph data files we used can all be found in the references of the paper.
+ The format of the graph data file should be SNAP, using '\t' as a separator for each row of vertex ids.
+ We use mask to represent the training set. The format of the mask file should be a text file consisting of 0s and 1s, with the number of lines equal to the number of vertices of the graph.
+ Note the modification of the data storage path in `0_bench_GNNA_GCN.py`.
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
> + Go to `GNNAdvisor/` directory.
>
> + `./0_bench_GNNA_GCN.py` to run baseline GNNAdvisor's 2-layer GCN model and report 100 epoch runtime for all evaluated datasets.

+ **Running GNNAdvisor**.
> + Checkout git branch to master.
>
>   ```shell
>   git checkout master
>   ```
>
> + Go to `GNNAdvisor/` directory. 
>
> + `./0_bench_GNNA_GCN.py` to run our GNNAdvisor's multi-layer GCN model and report 100 epoch runtime for all evaluated datasets in different train ratios and numbers of layers.
>
> + Set parameters `--partsize_model`  to choose different ways to decide `partsize` in backward process. There are three options for this parameter: `use_map`, `use_net` and `use_32`. `use_map` uses a linear regression equation, `use_net` uses a trained neural network and `use_32` uses a fixed constant of 32.
>
> +  Stand alone running `GNNA_main.py` with specified parameters.
> > + `--dataset`: the name of the dataset.
> > + `--dim`: the size of input embedding dimension, default: 96.
> > + `--hidden`: the size of hidden dimension, default: 16.
> > + `--classes`: the number of output classes, default: 22.
> > + `--layers`: the number of layers, default: 2.
> > + `--train_ratio`: the ratio of training set, default: 0.1
> > + `--partSize`: the size of neighbor-group in forward process, default: 32. 
> > + `--dimWorker`: the number of worker threads (**<=32**), default: 32.
> > + `--warpPerBlock`: the number of warp per block in forward process, default: 8, recommended: GCN: (8).
> > + `--sharedMem`: the shared memory size for each Stream-Multiprocessor on NVIDIA GPUs in . A reference for different GPU architecture and its shared memory size can be found at [here](https://en.wikipedia.org/wiki/CUDA).
> > + `--num_epoches`: the number of epoches for training, default: 100.
> > + `--loadFromTxt`: If this flag is `True`, it will load the graph TXT edge list, where each line is an `s1 d1`. default: `False` (load from `.npz` which is fast).
> > + `--enable_rabbit`: If this flag is `True`, it will be possible to use the rabbit-reordering routine. Otherwise, it will skip rabbit reordering for both **auto** and **manual** mode.

**Note**

> + We focus on the training evaluation of the GCNs, and the reported time per epoch only includes the GCN model forward and backward computation, excluding the data loading and some preprocessing. 


# Reference
+ [**Deep Graph Library**](https://github.com/dmlc/dgl) <br>
Wang, Minjie, et al. 
**Deep graph library: A graph-centric, highly-performant package for graph neural networks.**. *The International Conference on Learning Representations (ICLR), 2019.*
+ [**Pytorch Geometric**](https://github.com/rusty1s/pytorch_geometric) <br>
  Fey, Matthias, and Jan Eric Lenssen. 
  **Fast graph representation learning with PyTorch Geometric.** 
  *The International Conference on Learning Representations (ICLR), 2019.*
+ [**Rabbit Order**](https://github.com/araij/rabbit_order) <br>
  J. Arai, H. Shiokawa, T. Yamamuro, M. Onizuka, and S. Iwamura. 
  **Rabbit Order: Just-in-time Parallel Reordering for Fast Graph Analysis.** 
  *IEEE International Parallel and Distributed Processing Symposium (IPDPS), 2016.*