# -*- coding: utf-8 -*-
import argparse
import sys

sys.path.append('../../')

from script.evaluate.evaluate import predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ncr evaluate")
    PHASES = ['dev', 'test', 'train']
    parser.add_argument("--type", type=str, required=True, choices=PHASES,
                        help="datasets")
    args = parser.parse_args()

    save_path = f'result/ncr/{args.type}'
    max_seq_length = 2048
    split_token = '<问题>:\n'

    # eval_file_path = f"/data0/maqi/KGLQA-data/datasets/NCR/ncr_rocketqa_1400_instruct/{args.type}.jsonl"
    eval_file_path = f"/data0/maqi/KGLQA-data/datasets/NCR/Caption/ncr_caption_and_rel_new_instruct/{args.type}.jsonl"
    predict(eval_file_path, save_path, max_seq_length, split_token)
