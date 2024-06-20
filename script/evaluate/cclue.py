# -*- coding: utf-8 -*-
import argparse
import sys

sys.path.append('../../')

from script.evaluate.evaluate import predict

if __name__ == '__main__':
    """
    nohup python -u  cclue.py --type test > logs/cclue_test.log 2>&1 &
    nohup python -u  cclue.py --type dev > logs/cclue_dev.log 2>&1 &
    """
    parser = argparse.ArgumentParser(description="cclue evaluate")
    PHASES = ['dev', 'test', 'train']
    parser.add_argument("--type", type=str, required=False, choices=PHASES, default='test',
                        help="datasets")
    args = parser.parse_args()

    save_path = f'result/cclue_ft/{args.type}'

    # max_seq_length = 1400
    # split_token = '<question>:\n'
    # eval_file_path = f"/data0/maqi/KGLQA-data/datasets/CCLUE/cclue_instruct/{args.type}.jsonl"

    max_seq_length = 2048
    split_token = '<问题>:\n'
    eval_file_path = f"/data0/maqi/KGLQA-data/datasets/CCLUE/Caption/cclue_caption_and_rel_instruct/{args.type}.jsonl"

    # eval_file_path = f'/data0/maqi/KGLQA-data/datasets/CCLUE/random_select/cclue_random_1400_instruct/{args.type}.jsonl'
    # eval_file_path = f'/data0/maqi/KGLQA-data/datasets/CCLUE/random_select/cclue_chunk_1400_instruct/{args.type}.jsonl'
    # eval_file_path = f'/data0/maqi/KGLQA-data/datasets/CCLUE/cclue_instruct/{args.type}.jsonl'
    # eval_file_path = f'/data0/maqi/KGLQA-data/datasets/CCLUE/random_select/without_knowledge_chunk_instruct/{args.type}.jsonl'
    print(f"eval_file_path: {eval_file_path}")
    predict(eval_file_path, save_path, max_seq_length, split_token)
