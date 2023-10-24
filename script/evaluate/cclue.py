# -*- coding: utf-8 -*-
import argparse
import sys

sys.path.append('../../')

from script.evaluate.evaluate import predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cclue evaluate")
    PHASES = ['dev', 'test']
    parser.add_argument("--type", type=str, required=True, choices=PHASES,
                        help="datasets")
    args = parser.parse_args()

    save_path = 'result/cclue_ft'
    max_seq_length = 1400
    split_token = '<question>:\n'

    eval_file_path = f"/data0/maqi/KGLQA-data/datasets/CCLUE/cclue_instruct/{args.type}.jsonl"
    predict(eval_file_path, save_path, max_seq_length, split_token)