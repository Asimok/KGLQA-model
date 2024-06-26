import json
import os.path

import requests
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


def get_response(inputs, max_seq_length_, split_token_):
    # url = "http://219.216.64.127:7032/ablation_study"
    url = "http://219.216.64.116:7032/ablation_study"
    # url = 'http://219.216.64.231:27035/option1_quality_api'
    payload = json.dumps({
        "inputs": inputs,
        "max_seq_length": max_seq_length_,
        "split_token": split_token_
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    output = response.json()["response"]
    return output


def predict(eval_file_path, save_path, max_seq_length, split_token):
    # makedir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(eval_file_path, 'r') as f:
        samples = f.readlines()

    true_labels, pred_labels = [], []
    correct = 0
    for i, human_input in enumerate(tqdm(samples)):
        conv = json.loads(human_input.strip())
        req_input = conv["conversation"][0]["human"]
        label = conv["conversation"][0]["assistant"]
        true_labels.append(label)
        pred = get_response(req_input, max_seq_length, split_token)
        pred_labels.append(pred)
        # 格式化输出
        if label == pred:
            correct += 1
        print(
            f"\n{i + 1}\tlabel:{label}\tpred:{pred}\t{label == pred}\tacc:{correct / (i + 1)}\t")
    print('*' * 50)
    print(f"eval_file_path: {eval_file_path}")
    print(f"save_path: {save_path}")
    # 保留小数点后四位
    acc = accuracy_score(true_labels, pred_labels)
    print(f"acc:{acc:.4f}")
    print('*' * 50)
    metrics = classification_report(y_true=true_labels, y_pred=pred_labels, labels=['A', 'B', 'C', 'D'], digits=4)
    print(metrics)

    with open(f"{save_path}_eval.json", "w", encoding="utf-8") as f:
        f.write(json.dumps({"true_labels": true_labels, "pred_labels": pred_labels}, ensure_ascii=False, indent=4))
        f.write(metrics)


if __name__ == '__main__':
    #  测试 get_response
    req = 'Please determine if the following paragraphs can answer the question,output yes or no in json format:\nquestion:\nWhy is Si retirement so significant to the Space Exploration Team? \npassage:\n"Spaceman on a Spree" is a short story written by Mack Reynolds. The story follows the adventures of a retired space pilot named Seymour Pond after he receives a golden watch as a farewell gift from his colleagues. Despite feeling underwhelmed by the token gesture, Seymour sets out on a spree across the galaxy, encountering various challenges along the way. He meets new people, learns valuable lessons, and ultimately gains a sense of purpose and fulfillment in his post-retirement years.'
    res = get_response(req, 1536, '<question>:\n')
    print(res)
