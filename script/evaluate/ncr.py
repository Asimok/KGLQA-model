# -*- coding: utf-8 -*-
import argparse
import sys

sys.path.append('../../')

from script.evaluate.evaluate import predict

if __name__ == '__main__':
    """
    nohup python -u  ncr.py --type test > logs/ncr_test.log 2>&1 &
    nohup python -u  ncr.py --type dev > logs/ncr_dev.log 2>&1 &
    """
    parser = argparse.ArgumentParser(description="ncr evaluate")
    PHASES = ['dev', 'test', 'train']
    parser.add_argument("--type", type=str, choices=PHASES, default='dev',
                        help="datasets")
    args = parser.parse_args()

    save_path = f'result/ncr/{args.type}'
    max_seq_length = 1400
    split_token = '<question>:\n'
    # max_seq_length = 2048
    # split_token = '<问题>:\n'

    eval_file_path = f"/data0/maqi/KGLQA-data/datasets/NCR/ncr_rocketqa_1400_instruct/{args.type}.jsonl"
    # eval_file_path = f"/data0/maqi/KGLQA-data/datasets/NCR/Caption/ncr_caption_and_rel_instruct/{args.type}.jsonl"
    # eval_file_path = f'/data0/maqi/KGLQA-data/datasets/NCR/random_select/ncr_chunk_1400_instruct/{args.type}.jsonl'
    # eval_file_path = f'/data0/maqi/KGLQA-data/datasets/NCR/random_select/without_knowledge_chunk_instruct/{args.type}.jsonl'
    print(f"eval_file_path: {eval_file_path}")
    predict(eval_file_path, save_path, max_seq_length, split_token)
