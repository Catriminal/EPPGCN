#!/usr/bin/env python3
import os
import argparse
os.environ["PYTHONWARNINGS"] = "ignore"

parser = argparse.ArgumentParser()
parser.add_argument('--backsize_model', type=str, default='use_map', choices=['use_map', 'use_net', 'use_32'],  help="map or net or 32")
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

partsize_li = [32]          # only effective in manual model

dataset = [
        ('cora' 	        , 1433	    , 7 ,  ),        #
        ('citeseer'	        , 3703	    , 6 ,  ),        #
        ('pubmed'	        , 500	    , 3    ),        #
        # ('chameleon'	        , 2325	    , 5   ),    #  
        # ('actor'	        , 931	    , 5   ),        #
        # ('blog'	        , 512	    , 39   ),           #
        # ('reddit'	        , 602	    , 41   ),      
        ('youtube'	        , 64	    , 47   ),       #
        ('amazon'	        , 96	    , 22   ),       #

        ('corafull'	        , 64	    , 32   ),       #
        ('catalog'	        , 64	    , 32   ),       #
        # ('lj'	        , 64	    , 32   ),      
        ('twitter'	        , 64	    , 32   ),       #
        ('google'	        , 64	    , 32   ),       #
        ('dblp'	        , 64	    , 32   ),       #

        # ('ppi'	            , 50	    , 121 ),   

        # ('PROTEINS_full'             , 29       , 2) ,   
        # ('OVCAR-8H'                  , 66       , 2) , 
        # ('Yeast'                     , 74       , 2) ,
        # ('DD'                        , 89       , 2) ,
        # ('TWITTER-Real-Graph-Partial', 1323     , 2) ,   
        # ('SW-620H'                   , 66       , 2) ,
        
        # ( 'amazon0505'               , 96	, 22),
        # ( 'artist'                   , 100  , 12),
        # ( 'com-amazon'               , 96	, 22),
        # ( 'soc-BlogCatalog'	       	 , 128  , 39), 
        # ( 'amazon0601'  	         , 96	, 22), 
]

ratios = [0.1, 0.3, 0.5, 0.8]
# ratios = [0.3]

# backsize = { 'cora' :       {0.1 : [2, 1], 0.3 : [3, 2], 0.5 : [3, 3], 0.8 : [3, 3]},
#              'citeseer' :   {0.1 : [2, 1], 0.3 : [2, 2], 0.5 : [2, 2], 0.8 : [2, 2]},
#              'pubmed' :     {0.1 : [3, 2], 0.3 : [3, 3], 0.5 : [3, 6], 0.8 : [3, 6]},
#              'dblp' :       {0.1 : [6, 2], 0.3 : [6, 12], 0.5 : [6, 12], 0.8 : [6, 12]},
#              'youtube' :    {0.1 : [16, 14], 0.3 : [14, 14], 0.5 : [14, 13], 0.8 : [14, 14]},
#              'amazon' :     {0.1 : [6, 10], 0.3 : [6, 14], 0.5 : [6, 14], 0.8 : [6, 14]},
#              'corafull' :   {0.1 : [3, 4], 0.3 : [3, 6], 0.5 : [3, 6], 0.8 : [3, 6]},
#              'catalog' :    {0.1 : [28, 14], 0.3 : [28, 28], 0.5 : [28, 28], 0.8 : [28, 28]},
#              'twitter' :    {0.1 : [12, 12], 0.3 : [12, 12], 0.5 : [12, 28], 0.8 : [12, 28]},
#              'google' :     {0.1 : [8, 14], 0.3 : [8, 14], 0.5 : [8, 14], 0.8 : [8, 14]}}

backsize = { 'cora' :       {0.1 : [3, 1], 0.3 : [2, 3], 0.5 : [2, 3], 0.8 : [2, 2]},
             'citeseer' :   {0.1 : [1, 1], 0.3 : [3, 1], 0.5 : [3, 1], 0.8 : [3, 1]},
             'pubmed' :     {0.1 : [5, 1], 0.3 : [6, 3], 0.5 : [7, 3], 0.8 : [7, 7]},
             'dblp' :       {0.1 : [6, 1], 0.3 : [7, 3], 0.5 : [7, 5], 0.8 : [5, 5]},
             'youtube' :    {0.1 : [10, 8], 0.3 : [10, 7], 0.5 : [10, 7], 0.8 : [10, 10]},
             'amazon' :     {0.1 : [6, 9], 0.3 : [4, 7], 0.5 : [4, 15], 0.8 : [4, 4]},
             'corafull' :   {0.1 : [7, 1], 0.3 : [7, 3], 0.5 : [7, 5], 0.8 : [7, 7]},
             'catalog' :    {0.1 : [11, 11], 0.3 : [11, 11], 0.5 : [11, 11], 0.8 : [9, 10]},
             'twitter' :    {0.1 : [7, 11], 0.3 : [7, 11], 0.5 : [7, 7], 0.8 : [7, 7]},
             'google' :     {0.1 : [6, 6], 0.3 : [6, 7], 0.5 : [6, 10], 0.8 : [6, 6]}}


for partsize in partsize_li:
    for hid in hidden:
        for data, d, c in dataset:
            # print(data)
            for ratio in ratios:
                # print(ratio)
                dataDir = "/home/yc/data_scale_test/" + data
                l1_backsize = 0
                l2_backsize = 0
                if args.backsize_model == "use_net":
                    l1_backsize = backsize[data][ratio][0]
                    l2_backsize = backsize[data][ratio][1]
                elif args.backsize_model == "use_32":
                    l1_backsize = 32
                    l2_backsize = 32
                # dataDir = "../osdi-ae-graphs/"
                #--manual_mode {} --verbose_mode {} --enable_rabbit {} --loadFromTxt {} --dataDir {} --train_ratio {}
                command = "python /home/yc/OSDI21_AE-master/GNNAdvisor/GNNA_main.py --dataset {} --dim {} --hidden {} \
                            --classes {} --partSize {} --model {} --warpPerBlock {}\
                            --manual_mode {} --verbose_mode {} --enable_rabbit {} --loadFromTxt {} --dataDir {} --train_ratio {} --l1_backsize {}  --l2_backsize {}"
                command = command.format(data, d, hid, c, partsize, model, warpPerBlock,\
                                        manual_mode, verbose_mode, enable_rabbit, loadFromTxt, dataDir, ratio, l1_backsize, l2_backsize)		
                # print(command)
                                        # manual_mode, verbose_mode, enable_rabbit, loadFromTxt, dataDir)		
                # command = "python GNNA_main.py -loadFromTxt --dataset {} --partSize {} --dataDir {}".format(data, partsize, '/home/yuke/.graphs/orig')		 
                os.system(command)
