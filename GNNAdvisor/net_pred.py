#!/usr/bin/env python

import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from networks import Net
from torch.utils.data import random_split
import time
import GNNAdvisor as GNNA
import pandas as pd

conf_path = "/home/yc/OSDI21_AE-master/GNNAdvisor/net_pred_part_size"

def fileFetch(key):
    part_sizes = pd.read_csv(conf_path, names=["dataset", "part_size"], header=None, sep=' ')
    part_sizes = dict(zip(part_sizes['dataset'], part_sizes['part_size']))
    if key in part_sizes:
        return part_sizes[key]
    else:
        return None

def net_pred(id, edge_list, node_deg, valid_len, key):
    # data_time = 0.0
    # start = time.perf_counter()
    # local_start = time.perf_counter()
    dataset = []
    id = id[:valid_len]
    edge_list = edge_list[:valid_len]
    edge_index = torch.stack((id, edge_list), dim=0).long()
    # print("edge_index: {:.6f}".format(time.perf_counter() - local_start))
    # local_start = time.perf_counter()
    max_node = torch.max(edge_index)
    max_node += 1
    x = node_deg[:max_node]
    x = torch.reshape(x, (max_node, 1)).float()
    # print("x: {:.6f}".format(time.perf_counter() - local_start))
    # local_start = time.perf_counter()
    y = torch.zeros(1, dtype=torch.long)
    # print("y: {:.6f}".format(time.perf_counter() - local_start))
    
    # local_start = time.perf_counter()
    ginfo = GNNA.get_ginfo(node_deg, valid_len)

    # print(edge_index)
    # print(x)
    # print(y)
    # print(ginfo)
    data = Data(x, edge_index, None, y, ginfo=ginfo)
    dataset.append(data)
    # data_time += time.perf_counter() - start

    # load_time = 0.0
    # start = time.perf_counter()
    num_training = 0
    num_val = 0
    num_test = len(dataset)
    _, _, test_set = random_split(dataset, [num_training, num_val, num_test])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    device = torch.device('cuda')
    model = Net().to(device)

    model.load_state_dict(torch.load('/home/yc/param_opt/latest.pth', map_location={'cuda:1':'cuda:1'}))
    model = model.to(device)
    # load_time += time.perf_counter() - start

    # pred_time = 0.0
    # start = time.perf_counter()
    for data in test_loader:
        model.eval()
        data = data.to(device)
        out = model(data)
        pred = out.max(dim=1)[1]
    # pred_time += time.perf_counter() - start

    # print('data_time: {:.6f}'.format(data_time))
    # print('load_time: {:.6f}'.format(load_time))
    # print('pred_time: {:.6f}'.format(pred_time))

    re = int(pred.item()) + 1
    with open(conf_path, "a+") as f:
        f.write(key + ' ' + str(re) + '\n')

    return re

def get_net_back_part_size(id, edge_list, node_deg, valid_len, dataset, train_ratio, layer):
    key = dataset + '_b' + str(layer) + '_' + str(train_ratio)
    re = fileFetch(key)
    if re is None:
        re = net_pred(id, edge_list, node_deg, valid_len, key)
    
    return re