# -*- coding: utf-8 -*-
import argparse
import sys

sys.path.append('../../')
from script.evaluate.evaluate import predict

if __name__ == '__main__':
    """
    nohup python -u race.py --type middle > logs/race_middle.log 2>&1 &
    nohup python -u race.py --type high > logs/race_high.log 2>&1 &
    """
    parser = argparse.ArgumentParser(description="race evaluate")
    PHASES = ['middle', 'high']
    parser.add_argument("--type", type=str, required=True, choices=PHASES, default='middle',
                        help="datasets")
    args = parser.parse_args()

    save_path = f'result/race_{args.type}'
    max_seq_length = 2048
    split_token = '<question>:\n'

    eval_file_path = f"/data0/maqi/KGLQA-data/datasets/RACE/Caption/race_caption_and_rel_instruct/{args.type}_test.jsonl"
    # eval_file_path = f'/data0/maqi/KGLQA-data/datasets/RACE/instruct/race_{args.type}_test.jsonl'
    predict(eval_file_path, save_path, max_seq_length, split_token)


