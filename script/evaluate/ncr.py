# -*- coding: utf-8 -*-
import sys

sys.path.append('../../')

from script.evaluate.evaluate import predict

if __name__ == '__main__':
    save_path = 'result/ncr_1536_and_cclue'
    max_seq_length = 1536
    split_token = '<question>:\n'

    eval_file_path = "/data0/maqi/KGLTQA/datasets/NCR/ncr_firefly_format_1536/test.jsonl"
    predict(eval_file_path, save_path, max_seq_length, split_token)
