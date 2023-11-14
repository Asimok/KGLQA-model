# -*- coding: utf-8 -*-
import sys

sys.path.append('../../')

from script.evaluate.evaluate import predict

if __name__ == '__main__':
    save_path = 'result/quality'
    max_seq_length = 2048
    split_token = '<question>:\n'
    # eval_file_path = "/data0/maqi/KGLQA-data/datasets/QuALITY/Caption/quality_caption_and_rel_new_instruct/dev.jsonl"
    # eval_file_path = "/data0/maqi/KGLQA-data/datasets/QuALITY/Caption/quality_caption_and_rel_new2_instruct/dev.jsonl"
    eval_file_path = "/data0/maqi/KGLQA-data/datasets/QuALITY/Caption/quality_caption_and_rel_instruct/dev.jsonl"
    predict(eval_file_path, save_path, max_seq_length, split_token)
