#!/usr/bin/env python3
import torch
import math
import GNNAdvisor as GNNA
from param import *

class GNNAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, inputInfo, maskInfo, backInfo, isFirstIter, isBackModeNet):
        ctx.save_for_backward(X, weight)
        ctx.inputInfo = inputInfo
        ctx.backInfo = backInfo
        ctx.partSize, ctx.dimWorker, ctx.warpPerBlock = \
            inputInfo.partSize, inputInfo.dimWorker, inputInfo.warpPerBlock
        
        # print("[Foward]: {}\n{}\n{}\n{}\n{}".format(inputInfo.row_pointers, inputInfo.column_index, 
        #                                 inputInfo.degrees, inputInfo.partPtr, inputInfo.part2Node))    
        # print("[Foward]: partSize: {}, dimWorker: {}, warpPerBlock: {}".format(ctx.partSize, \
        #                                                     ctx.dimWorker, ctx.warpPerBlock))

        X_prime = GNNA.forward(X, weight, inputInfo.row_pointers, inputInfo.column_index, 
                                inputInfo.degrees, inputInfo.partPtr, inputInfo.part2Node, \
                                inputInfo.partSize, inputInfo.dimWorker, inputInfo.warpPerBlock)[0]
        
        if isFirstIter and not isBackModeNet:
            GNNA.mask_forward(inputInfo.part2Node, inputInfo.partPtr, inputInfo.column_index, maskInfo.src_mask, maskInfo.ngh_mask,
                                maskInfo.backEdgeMask, maskInfo.node_degs, maskInfo.num_layers, maskInfo.layer, maskInfo.blockx, maskInfo.blocky)

        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        X, weight = ctx.saved_tensors
        backInfo = ctx.backInfo

        d_input, d_weight = GNNA.ours_backward(d_output, X, weight, backInfo.id, backInfo.partPointer, 
                                        backInfo.edgeList, backInfo.degrees,
                                        backInfo.partSize, backInfo.numParts, backInfo.layer, backInfo.blockx, backInfo.blocky)
        return d_input, d_weight, None, None, None, None, None

class GCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

    def print_time(self):
        GNNA.print_time()

    def clear_time(self):
        GNNA.clear_time()

    def forward(self, X, inputInfo, maskInfo, backInfo, isFirstIter, isBackModeNet):
        '''
        @param:
        X:  the input tensor of the graph node embedding, shape: [n_nodes, n_dim].
        A:  the CSR node pointer of the graph, shape: [node, 1].
        edges: the CSR edge list of the graph, shape: [edge, 1].
        partitioin: for the graph with the part-based optimziation.
        '''
        return GNNAFunction.apply(X, self.weights, inputInfo, maskInfo, backInfo, isFirstIter, isBackModeNet)
