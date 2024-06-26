#!/bin/bash

nproc_per_node=1

#target=/data0/maqi/KGLQA-model/train_args/ablation_study/quality/option-2/without_knowledge_random.json
target=/data0/maqi/KGLQA-model/train_args/ncr/option-1/ncr_ft_alpaca_64k.json
#target=/data0/maqi/KGLQA-model/train_args/cclue/option-1/ncr_1_cclue_2_alpaca.json
#target=/data0/maqi/KGLQA-model/train_args/quality/option-2/race_ft_alpaca_1_quality_2.json

log_file=train.log
# 读取json文件中 output_dir 的值
output_dir=$(jq -r '.output_dir' ${target})
# 创建文件夹
if [ ! -d "${output_dir}" ];then
    mkdir -p "${output_dir}"
else
  # 询问是否删除 y删除 n不删除
  # shellcheck disable=SC2162
  read -p "是否删除 ${output_dir} 目录下所有文件？(y/n)" input
  if [ "${input}" = "y" ]; then
      # shellcheck disable=SC2115
      echo "已删除 ${output_dir} 目录下所有文件"
      # shellcheck disable=SC2115
      rm -rf "${output_dir}"/*
  else
      echo "保留 ${output_dir} 目录下所有文件"
  fi
fi




# 如果 nproc_per_node=1，那么就只使用第二块卡
if [ ${nproc_per_node} -eq 1 ];then
    export CUDA_VISIBLE_DEVICES=1
else
    export CUDA_VISIBLE_DEVICES=0,1
fi

#torchrun --nnodes 1 --nproc_per_node ${nproc_per_node} train_qlora.py --train_args_file ${target}
nohup torchrun --master_port 29501 --nnodes 1 --nproc_per_node ${nproc_per_node} /data0/maqi/KGLQA-model/train_qlora_flash_attn.py --train_args_file ${target} > "${output_dir}"/${log_file} 2>&1 &
# shellcheck disable=SC2046
echo $(pwd)/"${output_dir}"/"${log_file}"
# 打印进程号
echo 'pid:' $!