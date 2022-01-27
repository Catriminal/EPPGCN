#!/usr/bin/env python3
import os
import argparse
os.environ["PYTHONWARNINGS"] = "ignore"

parser = argparse.ArgumentParser()
parser.add_argument('--groupsize_model', type=str, default='regression_equation', choices=['regression_equation', 'SAGPG', 'fixed_value'])
args = parser.parse_args()


loadFromTxt = True         # whether to load data from a plain txt file.

model = 'gcn'
warpPerBlock = 8        # only effective in manual model
hidden = [16] 

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
                if args.groupsize_model == "SAGPG":
                    backMode = 'net'
                elif args.groupsize_model == "fixed_value":
                    backMode = 'constant'
                    backsize = 32
                
                command = "python /home/yc/OSDI21_AE-master/GNNAdvisor/gcn_main.py --dataset {} --dim {} --hidden {} \
                            --classes {} --layers {} --partSize {} --model {} --warpPerBlock {}\
                            --manual_mode {} --verbose_mode {} --loadFromTxt {} --dataDir {} --train_ratio {} --backsize_mode {} --backsize {}"
                command = command.format(data, d, hid, c, layer, partsize, model, warpPerBlock,\
                                        False, False, loadFromTxt, dataDir, ratio, backMode, backsize)		
                # print(command)
                os.system(command)
