# -*- coding: utf-8 -*-
import sys

sys.path.append('../../')

from script.evaluate.evaluate import predict

if __name__ == '__main__':
    save_path = 'result/cclue'
    max_seq_length = 1400
    split_token = '<question>:\n'

    eval_file_path = "/data0/maqi/KGLQA-data/datasets/CCLUE/cclue_instruct/test.jsonl"
    predict(eval_file_path, save_path, max_seq_length, split_token)
