#!/bin/bash

datasets=("cora" "citeseer" "pubmed" "youtube" "amazon" "corafull" "catalog" "twitter" "google" "dblp")
ratios=(0.1 0.3 0.5 0.8)
dims=(1433 3703 500 64 96 64 64 64 64 64)
classes=(7 6 3 47 22 32 32 32 32 32)
add="/home/yc/OSDI21_AE-master/GNNAdvisor/GNNA_main.py"
nvprof="/usr/local/cuda-10.1/bin/nvprof"
python="/home/yc/anaconda3/envs/gnnad/bin/python"
data_home="/home/yc/data_scale_test/"
re_home="/home/yc/OSDI21_AE-master/GNNAdvisor/nvprof_init_re/"
i=0
while [ $i -lt ${#datasets[@]} ]
do
    echo "${datasets[$i]}"
    for ratio in ${ratios[@]}
    do
        echo "$ratio"
        $nvprof --kernels "spmm" --metrics warp_execution_efficiency --log-file ${re_home}${datasets[$i]}_${ratio}.txt $python $add --dataset ${datasets[$i]} --dim ${dims[$i]} --classes ${classes[$i]} --dataDir ${data_home}${datasets[$i]} --train_ratio $ratio
    done
    let i++
done