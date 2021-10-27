import math 
import torch
import rabbit
import time
import numpy

def getBlockSize(dim):
    blockx = 32
    blocky = 2
    if dim < 8:
        blockx = 2
        blocky = 32
    elif dim >= 8 and dim < 32:
        blockx = 4
        blocky = 16
    elif dim >= 32 and dim < 96:
        blockx = 8
        blocky = 8
    elif dim >= 96 and dim < 192:
        blockx = 16
        blocky = 4
    elif dim >= 192 and dim < 512:
        blockx = 32
        blocky = 2
    return (blockx, blocky)

# package of input parameters
class inputProperty(object):
    def __init__(self, row_pointers=None, column_index=None, degrees=None,
                partSize=None, dimWorker=None, warpPerBlock=None, 
                sharedMem=None,
                hiddenDim=None,
                dataset_obj=None,
                enable_rabbit=False,
                manual_mode=True,
                verbose=False):
        
        if dataset_obj is None:
            raise ValueError("Dataset object MUST SET !!!")

        self.dataset_obj = dataset_obj

        self.row_pointers = row_pointers
        self.column_index = column_index
        self.degrees = degrees

        self.num_nodes = dataset_obj.num_nodes
        self.avgNodeDegree = dataset_obj.avg_degree
        self.avgEdgeSpan = dataset_obj.avg_edgeSpan

        self.partSize = partSize
        self.dimWorker = dimWorker
        self.warpPerBlock = warpPerBlock

        self.dimWorker_input = dimWorker
        self.dimWorker_hidden = dimWorker
        self.warpPerBlock_input = warpPerBlock
        self.warpPerBlock_hidden = warpPerBlock
        self.inputDim = dataset_obj.num_features
        self.hiddenDim = hiddenDim

        self.manual_mode = manual_mode
        self.enable_rabbit = enable_rabbit
        self.verbose_flag = verbose
        self.state_set_input = False
        self.reorder_status = False

        self.MAX_warpPerBlock = 8              
        self.share_memory = sharedMem * 0.4         
        self.gap_smem = 100

        self.partPtr = None
        self.part2Node = None

    def decider(self):
        '''
        Determine the performance-related parameter here.
        manual_mode: using user-specified parameters
        auto_mode:   determining the parameters according to the GPU resources and scheduling performance consideration.
        '''

        if self.manual_mode:
            if self.enable_rabbit:
                self.dataset_obj.reorder_flag = True
                self.dataset_obj.rabbit_reorder()
                self.reorder_status = True
                self.row_pointers = self.dataset_obj.row_pointers
                self.column_index = self.dataset_obj.column_index
            else:
                self.dataset_obj.reorder_flag = False
                self.reorder_status = False

            if self.verbose_flag:
                print("\n=> MANUAL Config Complete !!!\n")
        else:
            # Determine the neighbor partitioning.
            # self.partSize = int(self.avgNodeDegree)
            if self.avgNodeDegree < 4:
                self.partSize = 4
            elif self.avgNodeDegree >= 4 and self.avgNodeDegree < 16:
                self.partSize = 8
            elif self.avgNodeDegree >= 16 and self.avgNodeDegree < 64:
                self.partSize = 16
            elif self.avgNodeDegree >= 64 and self.avgNodeDegree < 256:
                self.partSize = 32
            elif self.avgNodeDegree >= 256 and self.avgNodeDegree < 512:
                self.partSize = 64

            est_shared = self.MAX_warpPerBlock * (self.partSize * 4 + self.inputDim * 4 + self.gap_smem * 4)/1e3
            if self.verbose_flag:
                print("input-layer shared memory (KB): {:.3f} ".format(est_shared))
            share_memory_input = min(est_shared, self.share_memory)
            if self.verbose_flag:
                print("input-layer updated (KB): {:.3f}".format(share_memory_input))

            est_shared = self.MAX_warpPerBlock * (self.partSize * 4 + self.hiddenDim + 4 * self.gap_smem)/1e3
            if self.verbose_flag:
                print("hidden-layer shared memory (KB): {:.3f}".format(est_shared))
            share_memory_hidden = min(est_shared, self.share_memory)
            if self.verbose_flag:
                print("hidden-layer updated (KB): {:.3f}".format(share_memory_hidden))

            # Determine the warpPerBlock for input and hidden layer.
            self.warpPerBlock_input = int(share_memory_input * 1e3 / (self.partSize * 4 + self.inputDim * 4))
            self.warpPerBlock_hidden = int(share_memory_hidden * 1e3 / (self.partSize * 4 + self.hiddenDim * 4))
            
            self.warpPerBlock_input = min(self.warpPerBlock_input, self.MAX_warpPerBlock)
            self.warpPerBlock_hidden = min(self.warpPerBlock_hidden, self.MAX_warpPerBlock)

            # Determine the dimWorker_input for input layer.
            if self.inputDim > 32:
                self.dimWorker_input = 32
            else:
                self.dimWorker_input = self.inputDim
            
            # Determine the dimWorker_hidden for hidden layer.
            if self.hiddenDim > 32:
                self.dimWorker_hidden = 32
            else:
                self.dimWorker_hidden = self.hiddenDim

            if self.enable_rabbit:
                # Determine whether to reorder a graph.
                if math.sqrt(self.avgEdgeSpan) > math.sqrt(self.num_nodes)/100:
                    self.dataset_obj.reorder_flag = True
                    self.reorder_status = True
                else:
                    self.dataset_obj.reorder_flag = False
                    self.reorder_status = False
                
                self.dataset_obj.rabbit_reorder()

            if self.verbose_flag:
                print("\n=> AUTO Decider Complete !!!\n")

    def set_input(self):
        '''
        Determine the performance-related parameter for input layer.
        Switch the parameter for input layer.
        '''
        self.dimWorker = self.dimWorker_input        
        self.warpPerBlock = self.warpPerBlock_input
        self.state_set_input = True

        return self        
    
    def set_hidden(self):
        '''
        Determine the performance-related parameter for hidden layer.
        Switch the parameter for hidden layer.
        '''
        self.dimWorker = self.dimWorker_hidden        
        self.warpPerBlock = self.warpPerBlock_hidden
        self.state_set_input = False
        return self   

    def print_param(self):
        if self.verbose_flag:
            if self.state_set_input:
                if self.manual_mode:
                    print("# manual INPUT partSize: {}".format(self.partSize))
                    print("# manual INPUT dimWorker: {}".format(self.dimWorker))
                    print("# manual INPUT warpPerBlock: {}".format(self.warpPerBlock))
                else:
                    print("# auto INPUT partSize: {}".format(self.partSize))
                    print("# auto INPUT dimWorker: {}".format(self.dimWorker))
                    print("# auto INPUT warpPerBlock: {}".format(self.warpPerBlock))
                    print("# auto INPUT reorder_flag: {}".format(self.reorder_status))
            else:
                if self.manual_mode:
                    print("# manual HIDDEN partSize: {}".format(self.partSize))
                    print("# manual HIDDEN dimWorker: {}".format(self.dimWorker))
                    print("# manual HIDDEN warpPerBlock: {}".format(self.warpPerBlock))
                else:
                    print("# auto HIDDEN partSize: {}".format(self.partSize))
                    print("# auto HIDDEN dimWorker: {}".format(self.dimWorker))
                    print("# auto HIDDEN warpPerBlock: {}".format(self.warpPerBlock))
                    print("# auto HIDDEN reorder_flag: {}".format(self.reorder_status))

class maskInputProperty(object):
    def __init__(self, src_mask=None, ngh_mask=None, backEdgeMask=None, node_degs=None, layer=None, dim=None):
        self.src_mask = src_mask
        self.ngh_mask = ngh_mask
        self.backEdgeMask = backEdgeMask
        self.node_degs = node_degs
        self.layer = layer
        self.blockx, self.blocky = getBlockSize(dim)
        self.dim = dim

class backInputProperty(object):
    def __init__(self, id=None, partPointer=None, edgeList=None, degrees=None,
                partSize=None, numParts=None, dim=None, layer=None):
        self.id = id
        self.partPointer = partPointer
        self.edgeList = edgeList
        self.degrees = degrees
        self.partSize = partSize
        self.numParts = numParts
        self.dim = dim
        self.layer = layer
        self.blockx, self.blocky = getBlockSize(dim)

    def reorder(self, numNodes):
        trans_time = 0.0
        trans_start = time.perf_counter()
        partPointer = self.partPointer.to(torch.device('cpu'))
        edgeList = self.edgeList.to(torch.device('cpu'))
        ids = self.id.to(torch.device('cpu'))
        trans_time += time.perf_counter() - trans_start

        partCount_start = time.perf_counter()
        partCount = (partPointer[1:] - partPointer[:-1]).tolist()
        partCount_time = time.perf_counter() - partCount_start
        partSize = self.partSize
        numParts = self.numParts

        partNumCount_start = time.perf_counter()
        partNumCount = [0] * (partSize + 1)
        partNumPos = [0] * (partSize + 1)
        for idx in range(numParts):
            cnt = partCount[idx]
            partNumCount[cnt] += 1
        partNumCount_time = time.perf_counter() - partNumCount_start
        print(partNumCount[0])
        partNumCount[0] = 0 # clear the part num whose part count is 0.

        partNumPos_start = time.perf_counter()
        for idx in range(1, len(partNumCount)):
            partNumCount[idx] = partNumCount[idx] + partNumCount[idx - 1]
            partNumPos[idx] = partNumCount[idx]
        partNumPos_time = time.perf_counter() - partNumPos_start
        
        sorted_start = time.perf_counter()
        sorted_partMap = [0] * partNumCount[-1]
        for idx in range(numParts):
            cnt = partCount[idx] - 1
            if cnt == -1:
                continue

            # if cnt >= partSize:
            #     print("cnt error: %d %d" % (cnt, partSize))
            #     exit(-1)
            # if partNumPos[cnt] >= numParts:
            #     print("pos error: %d %d" % (partNumPos[cnt], numParts))
            #     exit(-1)
            sorted_partMap[partNumPos[cnt]] = idx
            partNumPos[cnt] += 1
        sorted_time = time.perf_counter() - sorted_start

        edge_time = 0.0
        rabbit_time = 0.0
        edgeSort_time = 0.0
        newEdgeList_time = 0.0
        newPartPointer_time = 0.0
        reorder_start = time.perf_counter()
        new_partPointer = []
        new_partPointer.append(0)
        new_edgeList = []
        new_id = []
        interval = int(partSize / 2)
        for idx in range(1, partSize + 1, interval):
            startSize = idx
            endSize = min(idx + interval, partSize + 1)
            startNumOff = partNumCount[startSize - 1]
            endNumOff = partNumCount[endSize - 1]
            if endNumOff == startNumOff:
                continue
            edges = []
            
            edge_start = time.perf_counter()

            edges = rabbit.test_reorder(torch.IntTensor(sorted_partMap), partPointer, ids, edgeList,
                                        numNodes, startNumOff, endNumOff)

            # localPartNum = 0
            # for pos in range(startNumOff, endNumOff):
            #     partID = sorted_partMap[pos]
            #     partLen = partPointer[partID + 1] - partPointer[partID]
            #     partStart = partPointer[partID]
            #     id = int(ids[partStart])
            #     for off in range(partLen):
            #         dst = int(edgeList[partStart + off])
            #         edge = [numNodes + localPartNum, dst, id]
            #         edges.append(edge)
            #     localPartNum += 1
            edge_time += time.perf_counter() - edge_start
            
            # rabbit_start = time.perf_counter()
            # edges = rabbit.back_reorder(torch.IntTensor(edges))
            # rabbit_time += time.perf_counter() - rabbit_start

            edgeSort_start = time.perf_counter()
            edges = edges.numpy().tolist()
            edges.sort(key=lambda x : (x[0], x[1]))
            edgeSort_time += time.perf_counter() - edgeSort_start

            newEdgeList_start = time.perf_counter()
            last = edges[0][0]
            count = 0
            localPartCount = []
            for edge in edges:
                if edge[0] != last:
                    localPartCount.append(count)
                    count = 1
                    last = edge[0]
                else:
                    count += 1

                new_edgeList.append(edge[1])
                new_id.append(edge[2])
            localPartCount.append(count)
            newEdgeList_time += time.perf_counter() - newEdgeList_start

            newPartPointer_start = time.perf_counter()
            for count in localPartCount:
                new_partPointer.append(new_partPointer[-1] + count)
            newPartPointer_time += time.perf_counter() - newPartPointer_start
        reorder_time = time.perf_counter() - reorder_start

        trans_start = time.perf_counter()
        self.partPointer = torch.IntTensor(new_partPointer).to(torch.device('cuda'))
        self.edgeList = torch.IntTensor(new_edgeList).to(torch.device('cuda'))
        self.id = torch.IntTensor(new_id).to(torch.device('cuda'))
        trans_time = time.perf_counter() - trans_start
        self.numParts = partNumCount[-1]
        print("===================================================")
        print("trans_time: {:.6f}".format(trans_time))
        print("partCount_time: {:.6f}".format(partCount_time))
        print("partNumCount_time: {:.6f}".format(partNumCount_time))
        print("partNumPos_time: {:.6f}".format(partNumPos_time))
        print("sorted_time: {:.6f}".format(sorted_time))
        print("edge_time: {:.6f}".format(edge_time))
        print("rabbit_time: {:.6f}".format(rabbit_time))
        print("edgeSort_time: {:.6f}".format(edgeSort_time))
        print("newEdgeList_time: {:.6f}".format(newEdgeList_time))
        print("newPartPointer_time: {:.6f}".format(newPartPointer_time))
        print("reorder_time: {:.6f}".format(reorder_time))
        print("===================================================")

        