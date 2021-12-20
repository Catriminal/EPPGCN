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

from gnn_conv import *
from dataset import *

from net_pred import *

parser = argparse.ArgumentParser()
# Dataset related parameters.
parser.add_argument("--dataDir", type=str, default="../osdi-ae-graphs", help="the path to graphs")
parser.add_argument("--dataset", type=str, default='amazon0601', help="dataset")
parser.add_argument("--dim", type=int, default=96, help="input embedding dimension size")
parser.add_argument("--hidden", type=int, default=16, help="hidden dimension size")
parser.add_argument("--classes", type=int, default=22, help="output classes size")
parser.add_argument("--train_ratio", type=float, default=0.0, help="train mask ratio")

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
parser.add_argument('--enable_rabbit', type=str, choices=['True', 'False'], default='False', help="True: enable rabbit reordering, False, disable rabbit reordering, default: False (disable for both manual and auto mode).")
parser.add_argument('--loadFromTxt', type=str, choices=['True', 'False'], default='True', help="True: load the graph TXT edge list, False: load from .npy, default: False (load from npz fast)")
parser.add_argument('--single_spmm', type=str, choices=['True', 'False'], default='False', help="True: profile the single SpMM (neighbor aggregation) kernel for number epoches times")
parser.add_argument('--verify_spmm', type=str, choices=['True', 'False'], default='False', help="True: verify the output correctness of a single SpMM (neighbor aggregation) kernel against the CPU reference implementation.")

parser.add_argument("--l1_backsize", type=int, default=0, help="l1_back_part_size")
parser.add_argument("--l2_backsize", type=int, default=0, help="l2_back_part_size")

args = parser.parse_args()
# print()
# print()
print("||" + args.dataset + "   " + str(args.train_ratio))

partSize, dimWorker, warpPerBlock, sharedMem = args.partSize, args.dimWorker, args.warpPerBlock, args.sharedMem
manual_mode = args.manual_mode == 'True'
verbose_mode = args.verbose_mode == 'True'
enable_rabbit = args.enable_rabbit == 'True'
loadFromTxt = args.loadFromTxt == 'True'
single_spmm = args.single_spmm == 'True'
verify_spmm = args.verify_spmm == 'True'

# requires GPU for evaluation.
assert torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)

####################################
# loading data from files
####################################
if loadFromTxt:
    path = osp.join(args.dataDir, args.dataset + "_snap")
    dataset = custom_dataset(path, args.dataDir, args.train_ratio, args.dim, args.classes, load_from_txt=True, verbose=verbose_mode)
else:
    path = osp.join(args.dataDir, args.dataset+".npz")
    dataset = custom_dataset(path, args.dataDir, args.train_ratio, args.dim, args.classes, load_from_txt=False, verbose=verbose_mode)

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
                            hiddenDim=args.hidden, dataset_obj=dataset, enable_rabbit=enable_rabbit,
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

# inputInfo.partPtr = partPtr.int()
# inputInfo.part2Node = part2Node.int()
# inputInfo.reorder()

inputInfo.row_pointers  = inputInfo.row_pointers.to(device)
inputInfo.column_index  = inputInfo.column_index.to(device)
inputInfo.partPtr = partPtr.int().to(device)
inputInfo.part2Node  = part2Node.int().to(device)

l1_mask_input_prop = maskInputProperty(dataset.train_mask, dataset.ngh_mask, dataset.l2_edge_mask,
                                    dataset.l2_node_degs, 1, args.hidden)
l2_mask_input_prop = maskInputProperty(dataset.ngh_mask, dataset.empty_mask, dataset.l1_edge_mask, dataset.l1_node_degs,
                                    2, dataset.num_classes)

l1_back_input_prop = backInputProperty(degrees=degrees, dim=args.hidden, layer=1)
l2_back_input_prop = backInputProperty(degrees=degrees, dim=dataset.num_classes, layer=2)

####################################
# Verifing a single SpMM kernel
# against the CPU reference.
####################################
if verify_spmm:
    from unitest import *
    valid = Verification(args.hidden, \
                        inputInfo.row_pointers, inputInfo.column_index, degrees, \
                        inputInfo.partPtr, inputInfo.part2Node, \
                        partSize, dimWorker, warpPerBlock)
    valid.compute()
    valid.reference(dataset.edge_index, dataset.val, dataset.num_nodes)
    valid.compare()
    sys.exit(0)

####################################
# Profiling a single SpMM kernel
####################################
if single_spmm:
    from unitest import *
    valid = Verification(args.hidden, \
                        inputInfo.row_pointers, inputInfo.column_index, degrees, \
                        inputInfo.partPtr, inputInfo.part2Node, \
                        partSize, dimWorker, warpPerBlock)
    valid.profile_spmm(round=args.num_epoches)
    sys.exit(0)

####################################
# Building GNN model
####################################
if args.model == 'gcn':
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCNConv(dataset.num_features, args.hidden)
            self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        
        def print_time(self):
            self.conv1.print_time() # static method. only can count l1+l2 agg time
            
        def clear_time(self):
            self.conv1.clear_time()

        def forward(self, isFirstIter):
            x = dataset.x
            x = F.relu(self.conv1(x, inputInfo.set_input(), l1_mask_input_prop, l1_back_input_prop, isFirstIter))
            x = self.conv2(x, inputInfo.set_hidden(), l2_mask_input_prop, l2_back_input_prop, isFirstIter)
            return x
else:
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GINConv(dataset.num_features, args.hidden)
            self.conv2 = GINConv(args.hidden, args.hidden)
            self.conv3 = GINConv(args.hidden, args.hidden)
            self.conv4 = GINConv(args.hidden, args.hidden)
            self.conv5 = GINConv(args.hidden, dataset.num_classes)

        def forward(self):
            x = dataset.x
            x = F.relu(self.conv1(x, inputInfo.set_input()))
            x = F.relu(self.conv2(x, inputInfo.set_hidden()))
            x = F.relu(self.conv3(x, inputInfo.set_hidden()))
            x = F.relu(self.conv4(x, inputInfo.set_hidden()))
            x = self.conv5(x, inputInfo.set_hidden())
            return F.log_softmax(x, dim=1)

model, dataset = Net().to(device), dataset.to(device)
if verbose_mode:
    print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

####################################
# Define training function.
####################################

def buildBackPart():
    cpu_device = torch.device('cpu')
    l1_back_input_prop.id, l1_back_input_prop.edgeList, valid_len_tensor = GNNA.compact_back_edge(dataset.l1_edge_mask, inputInfo.column_index)
    l1_back_input_prop.partSize = net_pred(l1_back_input_prop.id.to(cpu_device), 
            l1_back_input_prop.edgeList.to(cpu_device),
            dataset.l1_node_degs.to(cpu_device),
            int(valid_len_tensor[0].item()))
    # print(type(num_edges))
    # print(type(l1_back_input_prop.partSize))
    # print(dataset.l1_node_degs)
    l1_back_input_prop.partPointer, num_parts_tensor = GNNA.split_back_part(dataset.l1_node_degs, num_edges, l1_back_input_prop.partSize)
    l1_back_input_prop.numParts = int(num_parts_tensor[0].item())

    l2_back_input_prop.id, l2_back_input_prop.edgeList, valid_len_tensor = GNNA.compact_back_edge(dataset.l2_edge_mask, inputInfo.column_index)
    l2_back_input_prop.partSize = net_pred(l2_back_input_prop.id.to(cpu_device), 
            l2_back_input_prop.edgeList.to(cpu_device),
            dataset.l2_node_degs.to(cpu_device),
            int(valid_len_tensor[0].item()))
    l2_back_input_prop.partPointer, num_parts_tensor = GNNA.split_back_part(dataset.l2_node_degs, num_edges, l2_back_input_prop.partSize)
    l2_back_input_prop.numParts = int(num_parts_tensor[0].item())

    # l1_back_input_prop.id, l1_back_input_prop.edgeList, l1_back_input_prop.partPointer, l1_back_info = \
    #     GNNA.build_back_part(dataset.l1_edge_mask, inputInfo.column_index, dataset.l1_node_degs, args.l1_backsize, l1_back_input_prop.dim)
    # l1_back_input_prop.partSize = int(l1_back_info[0].item())
    # l1_back_input_prop.numParts = int(l1_back_info[1].item())
    # # l1_back_input_prop.reorder(num_nodes)

    # l2_back_input_prop.id, l2_back_input_prop.edgeList, l2_back_input_prop.partPointer, l2_back_info = \
    #     GNNA.build_back_part(dataset.l2_edge_mask, inputInfo.column_index, dataset.l2_node_degs, args.l2_backsize, l2_back_input_prop.dim)

    # l2_back_input_prop.partSize = int(l2_back_info[0].item())
    # l2_back_input_prop.numParts = int(l2_back_info[1].item())
    # # l2_back_input_prop.reorder(num_nodes)
    # # print("build back part.")

for_time = 0.0
loss_time = 0.0
back_time = 0.0
build_time = 0.0
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
    x = model(isFirstIter)
    torch.cuda.synchronize()
    for_time += time.perf_counter() - start

    if(isFirstIter):
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