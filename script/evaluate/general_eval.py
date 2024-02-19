"""
通用格式推理
"""
import argparse

import sys

sys.path.append('../../')
import json

from tqdm import tqdm

from script.evaluate.evaluate import get_response

DICT_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}


def instruct_format(context, answer, query, options, conversation_id, captions=None):
    passage, answer, q, options = context, answer, query, options
    options = [option[4:] for option in options]

    prefix = (
        '阅读以下段落、摘要和问题，然后从选项中选择正确答案，答案应为A、B、C、D中的一个。\n\n')
    passage = f'<段落>:\n{passage}\n\n'
    question = f'<问题>:\n{query}\n\n'
    option = f'<选项>:\nA {options[0]}\nB {options[1]}\nC {options[2]}\nD {options[3]}\n\n'
    suffix = f"<答案>:\n"

    if captions is not None:
        caption = f'<摘要>:\n{captions}\n\n'
        prompt = ''.join([prefix, passage, caption, question, option, suffix])
    else:
        prompt = ''.join([prefix, passage, question, option, suffix])

    message = {"conversation_id": conversation_id,
               "category": "quality",
               "conversation": [
                   {
                       "human": prompt,
                       "assistant": answer
                   }]
               }
    return message


if __name__ == '__main__':
    """
    nohup python -u general_eval.py --proportion 0 > logs/0.log 2>&1 &
    nohup python -u general_eval.py --proportion 1 > logs/1.log 2>&1 &
    nohup python -u general_eval.py --proportion 0.1 > logs/0.1.log 2>&1 &
    nohup python -u general_eval.py --proportion 0.2 > logs/0.2.log 2>&1 &
    nohup python -u general_eval.py --proportion 0.3 > logs/0.3.log 2>&1 &
    nohup python -u general_eval.py --proportion 0.4 > logs/0.4.log 2>&1 &
    nohup python -u general_eval.py --proportion 0.5 > logs/0.5.log 2>&1 &
    nohup python -u general_eval.py --proportion 0.6 > logs/0.6.log 2>&1 &
    nohup python -u general_eval.py --proportion 0.7 > logs/0.7.log 2>&1 &
    nohup python -u general_eval.py --proportion 0.8 > logs/0.8.log 2>&1 &
    nohup python -u general_eval.py --proportion 0.9 > logs/0.9.log 2>&1 &
    
    """
    parser = argparse.ArgumentParser(description="eval")
    parser.add_argument("--proportion", type=float, required=False, default=None)
    args = parser.parse_args()
    proportion = args.proportion
    print('proportion: ', proportion)
    test_file = f'/data0/maqi/KGLQA-data/datasets/NCR/Caption/ncr_caption_and_rel_knowledge_select_analyze/test_{proportion}.jsonl'
    print('test_file: ', test_file)
    with open(test_file, 'r') as f:
        samples = f.readlines()
    pred_labels = []
    answers = {}
    correct = 0
    for i, human_input in enumerate(tqdm(samples)):
        conv = json.loads(human_input.strip())
        instruct_conv = instruct_format(context=conv['context'], answer='-1', query=conv['query'],
                                        options=[conv['option_0'], conv['option_1'], conv['option_2'],
                                                 conv['option_3']], conversation_id=None, captions=conv['captions'], )
        req_input = instruct_conv["conversation"][0]["human"]
        question_id = instruct_conv['conversation_id']
        pred = get_response(req_input, 2048, '<问题>:\n')
        print('\n', pred)
        pred_labels.append(pred)
        if pred == DICT_TO_LABEL[conv['label']]:
            correct += 1
        print('acc: ', correct / (i + 1))
    print('*' * 70)
    print('acc: ', correct / len(samples))
    print('*' * 70)
    print('pred done!')
