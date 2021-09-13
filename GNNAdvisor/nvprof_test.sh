#!/bin/bash

datasets=("cora" "citeseer" "pubmed" "chameleon" "actor" "blog" "youtube" "amazon" "corafull" "catalog" "twitter" "google")
ratios=(0.1 0.3 0.5 0.8)
dims=(1433 3703 500 2325 931 512 64 96 64 64 64 64)
classes=(7 6 3 5 5 39 47 22 32 32 32 32)
add="/home/yc/OSDI21_AE-master/GNNAdvisor/GNNA_main.py"
nvprof="/usr/local/cuda-10.1/bin/nvprof"
python="/home/yc/anaconda3/envs/gnnad/bin/python"
data_home="/home/yc/data_scale_test/"
re_home="/home/yc/OSDI21_AE-master/GNNAdvisor/nvprof_re/"
i=0
while [ $i -lt ${#datasets[@]} ]
do
    echo "${datasets[$i]}"
    for ratio in ${ratios[@]}
    do
        echo "$ratio"
        $nvprof --kernels "spmm" --metrics flop_count_sp_fma,warp_execution_efficiency --log-file ${re_home}${datasets[$i]}_${ratio}.txt $python $add --dataset ${datasets[$i]} --dim ${dims[$i]} --classes ${classes[$i]} --dataDir ${data_home}${datasets[$i]} --train_ratio $ratio
    done
    let i++
done