#!/usr/bin/env python
import sys
import time
import argparse
import os.path as osp
import numpy
import torch
import torch.nn.functional as F
from tqdm import *
from scipy.sparse import *

import GNNAdvisor as GNNA           # import GNNAdvisor

from gcn_conv import *
from dataset import *

from net_pred import *

parser = argparse.ArgumentParser()
# Dataset related parameters.
parser.add_argument("--dataDir", type=str, default="../osdi-ae-graphs", help="the path to graphs")
parser.add_argument("--dataset", type=str, default='amazon0601', help="dataset")
parser.add_argument("--dim", type=int, default=96, help="input embedding dimension size")
parser.add_argument("--hidden", type=int, default=16, help="hidden dimension size")
parser.add_argument("--classes", type=int, default=22, help="output classes size")
parser.add_argument("--layers", type=int, default=2, help="number of layers")
parser.add_argument("--train_ratio", type=float, default=0.1, help="train mask ratio")

# Model training related parameters.
parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gin'],  help="GCN or GIN")
parser.add_argument("--num_epoches", type=int, default=100, help="number of epoches for training, default=200")

# Manually set the performance related parameters
parser.add_argument("--partSize", type=int, default=32, help="neighbor-group size")
parser.add_argument("--dimWorker", type=int, default=32, help="number of worker threads (MUST < 32)")
parser.add_argument("--warpPerBlock", type=int, default=8, help="number of warp per block, recommended: GCN: 8, GIN: 2")
parser.add_argument("--sharedMem", type=int, default=100, help="shared memory size of each block (Quadro P6000 64(KB) sm_61), default=100(KB) for RTX3090 sm_86")

# Additional flags for studies.
parser.add_argument('--manual_mode', type=str, choices=['True', 'False'], default='False', help="True: use manual config, False: auto config, default: True")
parser.add_argument('--verbose_mode', type=str, choices=['True', 'False'], default='False', help="True: verbose mode, False: simple mode, default: False")
parser.add_argument('--loadFromTxt', type=str, choices=['True', 'False'], default='True', help="True: load the graph TXT edge list, False: load from .npy, default: False (load from npz fast)")
parser.add_argument('--single_spmm', type=str, choices=['True', 'False'], default='False', help="True: profile the single SpMM (neighbor aggregation) kernel for number epoches times")
parser.add_argument('--verify_spmm', type=str, choices=['True', 'False'], default='False', help="True: verify the output correctness of a single SpMM (neighbor aggregation) kernel against the CPU reference implementation.")

parser.add_argument("--backsize", type=int, default=0, help="backward part size of all layers")
# parser.add_argument("--l1_backsize", type=int, default=0, help="l1_back_part_size")
# parser.add_argument("--l2_backsize", type=int, default=0, help="l2_back_part_size")

parser.add_argument("--backsize_mode", type=str, default='net', choices=['map', 'net', 'constant'], help='map or net or constant')

args = parser.parse_args()
# print()
# print()
print("||" + args.dataset + "   " + str(args.train_ratio) + "   " + str(args.layers))

partSize, dimWorker, warpPerBlock, sharedMem = args.partSize, args.dimWorker, args.warpPerBlock, args.sharedMem
manual_mode = args.manual_mode == 'True'
verbose_mode = args.verbose_mode == 'True'
loadFromTxt = args.loadFromTxt == 'True'
single_spmm = args.single_spmm == 'True'
verify_spmm = args.verify_spmm == 'True'
backsize_mode = args.backsize_mode

# requires GPU for evaluation.
assert torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(1)

####################################
# loading data from files
####################################
if loadFromTxt:
    path = osp.join(args.dataDir, args.dataset + "_snap")
    dataset = custom_dataset(path, args.dataDir, args.train_ratio, args.dim, args.classes, args.layers, load_from_txt=True, verbose=verbose_mode)
else:
    path = osp.join(args.dataDir, args.dataset+".npz")
    dataset = custom_dataset(path, args.dataDir, args.train_ratio, args.dim, args.classes, args.layers, load_from_txt=False, verbose=verbose_mode)

num_nodes = dataset.num_nodes
num_edges = dataset.num_edges
column_index = dataset.column_index
row_pointers = dataset.row_pointers
degrees = dataset.degrees

####################################
# Building input property profile.
####################################
inputInfo = inputProperty(row_pointers, column_index, degrees, 
                            partSize, dimWorker, warpPerBlock, sharedMem,
                            hiddenDim=args.hidden, dataset_obj=dataset,
                            manual_mode=manual_mode, verbose=verbose_mode)

####################################
# Decider for parameter selection.
####################################
inputInfo.decider()
# print(inputInfo.dimWorker_input)
# print(inputInfo.dimWorker_hidden)
# print(inputInfo.warpPerBlock_input)
# print(inputInfo.warpPerBlock_hidden)

inputInfo = inputInfo.set_input()
if verbose_mode:
    print('----------------------------')
    inputInfo.print_param()
    print()

inputInfo = inputInfo.set_hidden()
if verbose_mode:
    inputInfo.print_param()
    print()
    print('----------------------------')
# sys.exit(0)

####################################
# Building neighbor partitioning.
####################################
start = time.perf_counter()
partPtr, part2Node = GNNA.build_part(inputInfo.partSize, inputInfo.row_pointers)
build_neighbor_parts = time.perf_counter() - start
if verbose_mode:
    print("# Build nb_part (s): {:.3f}".format(build_neighbor_parts))

inputInfo.row_pointers  = inputInfo.row_pointers.to(device)
inputInfo.column_index  = inputInfo.column_index.to(device)
inputInfo.partPtr = partPtr.int().to(device)
inputInfo.part2Node  = part2Node.int().to(device)

mask_input_props = []
for i in range(0, args.layers):
    src_mask = dataset.ngh_masks[i - 1] if i != 0 else dataset.train_mask
    dst_mask = dataset.ngh_masks[i] if i < args.layers - 1 else dataset.empty_mask
    back_edge_mask = dataset.edge_masks[args.layers - 1 - i]
    node_deg = dataset.node_degs[args.layers - 1 - i]
    layer = i + 1
    dim = args.hidden if i < args.layers - 1 else dataset.num_classes

    mask_input_props.append(maskInputProperty(src_mask, dst_mask, back_edge_mask, node_deg, args.layers, layer, dim))

back_input_props = []
for i in range(0, args.layers):
    dim = args.hidden if i < args.layers - 1 else dataset.num_classes
    layer = i + 1
    back_input_props.append(backInputProperty(degrees=degrees, dim=dim, layer=layer))

def mask_forward(inputInfo, maskInfo):
    GNNA.mask_forward(inputInfo.part2Node, inputInfo.partPtr, inputInfo.column_index, maskInfo.src_mask, maskInfo.ngh_mask,
                                maskInfo.backEdgeMask, maskInfo.node_degs, maskInfo.num_layers, maskInfo.layer, maskInfo.blockx, maskInfo.blocky)

def buildBackPart():
    cpu_device = torch.device('cpu')
    for i in range(0, args.layers):
        id, edgeList, valid_len_tensor = GNNA.compact_back_edge(dataset.edge_masks[i], inputInfo.column_index)
        partSize = 0
        if backsize_mode == 'net':
            partSize = get_net_back_part_size(id.to(cpu_device), edgeList.to(cpu_device),
                    dataset.node_degs[i].to(cpu_device),
                    int(valid_len_tensor[0].item()),
                    args.dataset,
                    args.train_ratio,
                    i + 1)
        elif backsize_mode == 'map':
            partSize = GNNA.get_map_back_part_size(dataset.node_degs[i], int(valid_len_tensor[0].item()))
        else:
            partSize = args.backsize
        partPointer, num_parts_tensor = GNNA.split_back_part(dataset.node_degs[i], num_edges, partSize)
        back_input_props[i].fillProperty(id, partPointer, edgeList, partSize, int(num_parts_tensor[0].item()))

build_time = 0.0
if backsize_mode == 'net':
    start = time.perf_counter()
    for i in range(0, args.layers):
        mask_forward(inputInfo, mask_input_props[i])
    buildBackPart()
    build_time += time.perf_counter() - start


####################################
# Building GCN model
####################################

class Net(torch.nn.Module):
    def __init__(self, num_layers):
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        for i in range(0, num_layers):
            input_dim = args.hidden if i != 0 else dataset.num_features
            output_dim = args.hidden if i < num_layers - 1 else dataset.num_classes
            self.convs.append(GCNConv(input_dim, output_dim))
    
    def print_time(self):
        self.convs[0].print_time() # static method.
        
    def clear_time(self):
        self.convs[0].clear_time() # static method.

    def forward(self, isFirstIter, isBackModeNet):
        x = dataset.x
        for i in range(0, self.num_layers):
            if i == 0:
                x = F.relu(self.convs[i](x, inputInfo.set_input(), mask_input_props[i], back_input_props[i], isFirstIter, isBackModeNet))
            elif i < self.num_layers - 1:
                x = F.relu(self.convs[i](x, inputInfo.set_hidden(), mask_input_props[i], back_input_props[i], isFirstIter, isBackModeNet))
            else:
                x = self.convs[i](x, inputInfo.set_hidden(), mask_input_props[i], back_input_props[i], isFirstIter, isBackModeNet)
        return x

model, dataset = Net(args.layers).to(device), dataset.to(device)
if verbose_mode:
    print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

####################################
# Define training function.
####################################

for_time = 0.0
loss_time = 0.0
back_time = 0.0
other_time = 0.0

def train(isFirstIter):
    global for_time
    global loss_time
    global back_time
    global build_time
    # global other_time
    # torch.cuda.synchronize()
    # start = time.perf_counter()

    model.train()
    optimizer.zero_grad()
    # torch.cuda.synchronize()
    # other_time += time.perf_counter() - start

    # torch.cuda.synchronize()
    start = time.perf_counter()
    x = model(isFirstIter, backsize_mode == 'net')
    torch.cuda.synchronize()
    for_time += time.perf_counter() - start

    if(isFirstIter and backsize_mode != 'net'):
        start = time.perf_counter()
        buildBackPart()
        build_time += time.perf_counter() - start

    # torch.cuda.synchronize()
    start = time.perf_counter()
    x = F.log_softmax(x, dim=1)
    loss = F.nll_loss(x[dataset.train_mask], dataset.y[dataset.train_mask])
    torch.cuda.synchronize()
    loss_time += time.perf_counter() - start

    # torch.cuda.synchronize()
    start = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    back_time += time.perf_counter() - start

    # torch.cuda.synchronize()
    # start = time.perf_counter()
    optimizer.step()
    # torch.cuda.synchronize()
    # other_time += time.perf_counter() - start

if __name__ == '__main__':
    # dry run

    for i in range(10):
        train(i == 0)
    # exit(0)
    for_time = 0.0
    loss_time = 0.0
    back_time = 0.0
    # other_time = 0.0
    model.clear_time()
    torch.cuda.synchronize()
    start_train = time.perf_counter()
    for _ in range(1, args.num_epoches + 1):
    # for _ in tqdm(range(1, args.num_epoches + 1)):
        train(False)
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_train
    model.print_time()
    print('for_time: {:.6f}'.format(for_time))
    print('loss_time: {:.6f}'.format(loss_time))
    print('back_time: {:.6f}'.format(back_time))
    print('build_time: {:.6f}'.format(build_time))
    print('Time: {:.6f}'.format(total_time))
    # print('Time (ms): {:.3f}'.format(total_time*1e3/args.num_epoches))
    print()