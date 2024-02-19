import os

import collections
import random
import sys

sys.path.append('../../')
import json

from tqdm import tqdm

from script.evaluate.evaluate import get_response

DICT_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}


def instruct_format(context, answer, query, options, conversation_id, captions=None):
    passage, answer, q, options = context, answer, query, options
    options = [option[4:] for option in options]

    prefix = ('Read the following passage and questions, then choose the right answer from options, the answer '
              'should be one of A, B, C, D.\n\n')
    passage = f'<passage>:\n{passage}\n\n'
    question = f'<question>:\n{q}\n\n'
    option = f'<options>:\nA {options[0]}\nB {options[1]}\nC {options[2]}\nD {options[3]}\n\n'
    suffix = f"<answer>:\n"
    if captions is not None:
        caption = f'<summary>:\n{captions}\n\n'
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


def predict(phase_):
    with open(test_file_base + f'/test_{phase_}.jsonl', 'r') as f:
        samples = f.readlines()
    pred_labels = []
    correct = 0
    for i, human_input in enumerate(tqdm(samples)):
        conv = json.loads(human_input.strip())
        instruct_conv = instruct_format(context=conv['context'], answer='-1', query=conv['query'],
                                        options=[conv['option_0'], conv['option_1'], conv['option_2'],
                                                 conv['option_3']], conversation_id=1, captions=None)
        req_input = instruct_conv["conversation"][0]["human"]
        pred = get_response(req_input, max_seq_length_=phase_, split_token_='<question>:\n')
        if pred == DICT_TO_LABEL[conv['label']]:
            correct += 1
        print('\npred:', pred, '\tacc:', correct / (i + 1))
        pred_labels.append(pred)

    # 统计 pred_labels 中元素个数
    answer_dict_ = collections.Counter(pred_labels)
    print(answer_dict_)
    print(f'{phase_}-final acc: ', correct / len(samples))
    return phase_, correct / len(samples)


if __name__ == '__main__':
    """
    nohup python -u len_analyze_ncr.py > logs/len_analyze_ncr.log 2>&1 &
    {'1200': 0.5439137134052389, '1600': 0.5439137134052389, '400': 0.5003852080123267, '600': 0.5281201848998459, '800': 0.5365947611710323, '1000': 0.5450693374422187}
    """
    phase = [1400]
    test_file_base = f'/data0/maqi/KGLQA-data/datasets/NCR/len_analyze'
    out_file = '/data0/maqi/KGLQA-data/datasets/NCR/len_analyze/result.json'
    try:
        with open(out_file, 'r') as f:
            answer_dict = json.load(f)
        print('load his answer dict:', answer_dict)
    except Exception as e:
        print(str(e))
        answer_dict = {}
        print('create new answer dict')

    for p in phase:
        try:
            phase_, acc = predict(p)
            answer_dict[phase_] = acc
        except Exception as e:
            print(str(e))
    print(answer_dict)
    with open(out_file, 'w') as f:
        json.dump(answer_dict, f)
    print('save to ', out_file)
    print('done!')
