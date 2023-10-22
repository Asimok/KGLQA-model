# -*- coding: utf-8 -*-
import sys

sys.path.append('../../')

import json
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from script.evaluate.evaluate import get_response

if __name__ == '__main__':
    model = 'result/race_middle'
    max_seq_length = 2048
    split_token = '\nquestion:\n'

    with open("/data0/maqi/KGLTQA/datasets/race/race_middle_test.jsonl", 'r') as f:
        # with open("/data0/maqi/KGLTQA/datasets/race/race_high_test.jsonl", 'r') as f:
        samples = f.readlines()

    true_labels, pred_labels = [], []
    for i, human_input in enumerate(tqdm(samples)):
        conv = json.loads(human_input.strip())
        req_input = conv["conversation"][0]["human"]
        label = conv["conversation"][0]["assistant"]
        true_labels.append(label)
        pred = get_response(req_input, max_seq_length, split_token)
        pred_labels.append(pred)
        # 格式化输出
        print(f"\n{i + 1}\tlabel:{label}\tpred:{pred}\t{label == pred}\tacc:{accuracy_score(true_labels, pred_labels)}")

    print(classification_report(true_labels, pred_labels, digits=4))

    with open(f"{model}_eval.json", "w", encoding="utf-8") as f:
        f.write(json.dumps({"true_labels": true_labels, "pred_labels": pred_labels}, ensure_ascii=False, indent=4))
        f.write(classification_report(true_labels, pred_labels, digits=4))
