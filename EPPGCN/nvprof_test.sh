#!/bin/bash

datasets=("cora" "citeseer" "pubmed" "youtube" "amazon" "corafull" "catalog" "twitter" "google" "dblp")
# datasets=("catalog")
ratios=(0.1 0.3 0.5 0.8)
dims=(1433 3703 500 64 96 64 64 64 64 64)
classes=(7 6 3 47 22 32 32 32 32 32)
# enum best part size
# l1_sizes=(2 2 3 2 1 3 5 4 4 6 6 7 8 10 10 10 3 4 4 4 5 7 7 7 25 26 21 27 11 11 9 10 5 6 6 6 6 12 6 5)
# l2_sizes=(2 2 3 2 1 3 1 1 4 3 5 5 8 7 7 7 9 10 15 3 1 2 3 7 18 12 19 22 11 11 7 9 6 6 3 3 1 3 7 5)

# net part size
l1_sizes=(2 2 2 2 2 2 2 2 5 7 7 7 10 6 6 6 6 6 4 4 7 7 7 7 11 11 21 27 11 11 27 16 6 6 6 6 7 7 7 7)
l2_sizes=(1 2 2 2 1 1 2 2 1 3 5 7 8 7 10 10 9 7 6 6 1 3 5 7 11 11 11 21 11 11 11 27 8 10 6 6 1 3 5 7)
add="/home/yc/OSDI21_AE-master/GNNAdvisor/GNNA_main.py"
nvprof="/usr/local/cuda-10.1/bin/nvprof"
python="/home/yc/anaconda3/envs/gnnad/bin/python"
data_home="/home/yc/data_scale_test/"
re_home="/home/yc/OSDI21_AE-master/GNNAdvisor/nvprof_warp_re/"
i=0
while [ $i -lt ${#datasets[@]} ]
do
    echo "${datasets[$i]}"
    j=0
    while [ $j -lt ${#ratios[@]} ]
    do
        echo "${ratios[$j]}"
        k=$(( $i * 4 + $j ))
        $nvprof --kernels "ours" --metrics warp_execution_efficiency --log-file ${re_home}${datasets[$i]}_${ratios[$j]}.txt $python $add --dataset ${datasets[$i]} --dim ${dims[$i]} --classes ${classes[$i]} --dataDir ${data_home}${datasets[$i]} --train_ratio ${ratios[$j]} --l1_backsize ${l1_sizes[$k]} --l2_backsize ${l2_sizes[$k]}
        let j++
    done
    let i++
done