#!/bin/bash

target=/data0/maqi/Firefly/train_args/quality/option1-quality-and-race-4096.json
# 读取json文件中 output_dir 的值
output_dir=$(jq -r '.output_dir' ${target})
# 创建文件夹
if [ ! -d "${output_dir}" ];then
    mkdir "${output_dir}"
else
    rm -rf "${output_dir}"
    mkdir "${output_dir}"
fi
log_file=option1-quality-and-race-4096

nproc_per_node=2

# 如果 nproc_per_node=1，那么就只使用第二块卡
if [ ${nproc_per_node} -eq 1 ];then
    export CUDA_VISIBLE_DEVICES=1
else
    export CUDA_VISIBLE_DEVICES=0,1
fi

nohup torchrun --nnodes 1 --nproc_per_node ${nproc_per_node} train_qlora.py --train_args_file ${target} > ${output_dir}/${log_file}.log 2>&1 &
# shellcheck disable=SC2046
echo $(pwd)/"${output_dir}"/${log_file}.log