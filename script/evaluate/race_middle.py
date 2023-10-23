# -*- coding: utf-8 -*-
import sys

sys.path.append('../../')

from script.evaluate.evaluate import predict

sys.path.append('../../')

if __name__ == '__main__':
    save_path = 'result/race_middle'
    max_seq_length = 2048
    split_token = '<question>:\n'

    eval_file_path = "/data0/maqi/KGLQA-data/datasets/RACE/race_middle_test.jsonl"
    predict(eval_file_path, save_path, max_seq_length, split_token)
