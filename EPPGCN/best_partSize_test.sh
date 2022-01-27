#!/bin/bash

datasets=(cora citeseer pubmed youtube amazon corafull catalog twitter google dblp)
dims=(1433 3703 500 64 96 64 64 64 64 64)
classes=(7 6 3 47 22 32 32 32 32 32)
# datasets=(cora youtube)
# dims=(1433 64)
# classes=(7 47)
ratios=(0.1 0.3 0.5 0.8)
# ratios=(0.8)
gnnad="/home/yc/OSDI21_AE-master/GNNAdvisor/gcn_main.py"
# dataset
# dim
# classes
# dataDir
# train_ratio
# l1_backsize
# l2_backsize

i=0
while [ $i -lt ${#datasets[@]} ]
do
    data=${datasets[$i]}
    dim=${dims[$i]}
    class=${classes[$i]}
    dataDir="/home/yc/data_scale_test/${data}"
    echo "$data"
    for ratio in ${ratios[@]}
    do
        echo "$ratio"
        min_l1_agg=10000.0
        best_l1_part=0
        min_l2_agg=10000.0
        best_l2_part=0
        for partsize in {1..16}
        do
            echo "part_size: $partsize"
            # echo "$gnnad --dataset $data --dim $dim --classes $class --dataDir $dataDir --train_ratio $ratio --l1_backsize $partsize --l2_backsize $partsize"
            re=`$gnnad --dataset $data --dim $dim --classes $class --dataDir $dataDir --train_ratio $ratio --l1_backsize $partsize --l2_backsize $partsize`
            
            l1_agg=`echo $re | awk '{match($0, /l1_back_agg_time: ([0-9]+.[0-9]+)/, a);print a[1]}'`
            l2_agg=`echo $re | awk '{match($0, /l2_back_agg_time: ([0-9]+.[0-9]+)/, a);print a[1]}'`
            echo "l1_agg: $l1_agg"
            echo "l2_agg: $l2_agg"
            if [ `echo "$l1_agg < $min_l1_agg"|bc` -eq 1 ] ; then
                min_l1_agg=$l1_agg
                best_l1_part=$partsize
            fi
            
            if [ `echo "$l2_agg < $min_l2_agg"|bc` -eq 1 ] ; then
                min_l2_agg=$l2_agg
                best_l2_part=$partsize
            fi
        done
        echo -e "dataset\t${data}_${ratio}"
        echo -e "best_l1_part\t$best_l1_part"
        echo -e "best_l2_part\t$best_l2_part"
    done
    let i++
done
echo "for-----------------------------------"
i=0
while [ $i -lt ${#datasets[@]} ]
do
    data=${datasets[$i]}
    dim=${dims[$i]}
    class=${classes[$i]}
    dataDir="/home/yc/data_scale_test/${data}"
    echo "$data"
    # for ratio in ${ratios[@]}
    # do
    ratio=0.3
    echo "$ratio"
    min_for_agg=10000.0
    best_for_part=0
    for partsize in {1..32}
    do
        echo "part_size: $partsize"
        re=`$gnnad --dataset $data --dim $dim --classes $class --dataDir $dataDir --train_ratio $ratio --partSize $partsize`
        # echo "$gnnad --dataset $data --dim $dim --classes $class --dataDir $dataDir --train_ratio $ratio --l1_backsize $partsize --l2_backsize $partsize"
        for_agg=`echo $re | awk '{match($0, /for_agg_time: ([0-9]+.[0-9]+)/, a);print a[1]}'`
        echo "for_agg: $for_agg"
        if [ `echo "$for_agg < $min_for_agg"|bc` -eq 1 ] ; then
            min_for_agg=$for_agg
            best_for_part=$partsize
        fi
    done
    echo -e "dataset\t${data}_${ratio}"
    echo -e "best_for_part\t$best_for_part"
    # done
    let i++
done