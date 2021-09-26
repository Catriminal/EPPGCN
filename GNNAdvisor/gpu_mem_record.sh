#!/bin/bash

total=$1
interval=0.1
cur=0
max=0
while [ $cur -lt $total ]
do
    info=`nvidia-smi | grep 16280  | awk '{match($0,/([0-9]+)MiB \//, a);print a[1]}'`
    arr=(${info//\n/ })
    # echo ${arr[0]}
    if [ $max -lt ${arr[0]} ]
    then
        max=${arr[0]}
    fi
    sleep $interval
    cur=$[$cur+1]
    # echo $cur
done
echo "max memory-usage is $max"