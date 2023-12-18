#!/bin/bash
# 设置需要查询的GPU
gpu_id=1
# 设置需要的显存
need=35000
flag=1
while [ "$flag" -eq 1 ];
do
    free_m=$(nvidia-smi --id="$gpu_id" --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print $1}')
    echo "$free_m"
    if [ "$free_m" -gt $need ]; then
        echo "GPU $gpu_id is free"
        # 执行脚本
        sh /data0/maqi/KGLQA-model/script/run/run_sft.sh
        flag=0
        break
    fi
    sleep 5
done
