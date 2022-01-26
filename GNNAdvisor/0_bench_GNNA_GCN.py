#!/usr/bin/env python3
import os
import argparse
os.environ["PYTHONWARNINGS"] = "ignore"

parser = argparse.ArgumentParser()
parser.add_argument('--partsize_model', type=str, default='use_map', choices=['use_map', 'use_net', 'use_32'],  help="map or net or 32")
args = parser.parse_args()

run_GCN = True              # whether to run GCN model. 
enable_rabbit = False        # whether to enable rabbit reordering in auto and manual mode.
manual_mode = False         # whether to use the manually configure the setting.
verbose_mode = False         # whether to printout more information such as the layerwise parameter.
loadFromTxt = True         # whether to load data from a plain txt file.

if run_GCN:
    model = 'gcn'
    warpPerBlock = 8        # only effective in manual model
    hidden = [16] 
else:
    model = 'gin'
    warpPerBlock = 2        # only effective in manual model 2 for citeseer 6 for remaining datasets
    hidden = [64] 		

dataset = [
        ('cora' 	        , 1433	    , 7 ,  ),        #
        ('citeseer'	        , 3703	    , 6 ,  ),        #
        ('pubmed'	        , 500	    , 3    ),        #
        ('youtube'	        , 64	    , 47   ),       #
        ('amazon'	        , 96	    , 22   ),       #

        ('corafull'	        , 64	    , 32   ),       #
        ('catalog'	        , 64	    , 32   ),       #
        ('twitter'	        , 64	    , 32   ),       #
        ('google'	        , 64	    , 32   ),       #
        ('dblp'	        , 64	    , 32   ),       #

]

ratios = [0.1, 0.3, 0.5, 0.8]
# ratios = [0.1]

# layers = [3]
layers = [3, 4, 5, 6]

partsize = 0

for hid in hidden:
    for data, d, c in dataset:
        # print(data)
        for ratio in ratios:
            # print(ratio)
            for layer in layers:
                dataDir = "/home/yc/data_scale_test/" + data
                backsize = 0
                backMode = 'map'
                if args.partsize_model == "use_net":
                    backMode = 'net'
                elif args.partsize_model == "use_32":
                    backMode = 'constant'
                    backsize = 32
                
                command = "python /home/yc/OSDI21_AE-master/GNNAdvisor/GNNA_main.py --dataset {} --dim {} --hidden {} \
                            --classes {} --layers {} --partSize {} --model {} --warpPerBlock {}\
                            --manual_mode {} --verbose_mode {} --enable_rabbit {} --loadFromTxt {} --dataDir {} --train_ratio {} --backsize_mode {} --backsize {}"
                command = command.format(data, d, hid, c, layer, partsize, model, warpPerBlock,\
                                        manual_mode, verbose_mode, enable_rabbit, loadFromTxt, dataDir, ratio, backMode, backsize)		
                # print(command)
                os.system(command)
