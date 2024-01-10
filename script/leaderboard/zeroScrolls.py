import os

import collections
import sys

sys.path.append('../../')
import json

from tqdm import tqdm

from script.evaluate.evaluate import get_response


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


# test_file = '/data0/maqi/KGLQA-data/datasets/QuALITY/quality_rocketqa_2048/test.jsonl'
test_file = '/data0/maqi/KGLQA-data/datasets/QuALITY/Caption/quality_caption_and_rel/test.jsonl'
out_file = '/data0/maqi/KGLQA-data/datasets/QuALITY/predictions/zero_scrolls/kb_alpaca_pred.json'
if not os.path.exists(os.path.dirname(out_file)):
    os.makedirs(os.path.dirname(out_file))

with open(test_file, 'r') as f:
    samples = f.readlines()
pred_labels = []
answers = {}
correct = 0
for i, human_input in enumerate(tqdm(samples)):
    conv = json.loads(human_input.strip())
    instruct_conv = instruct_format(context=conv['context'], answer='-1', query=conv['query'],
                                    options=[conv['option_0'], conv['option_1'], conv['option_2'],
                                             conv['option_3']], conversation_id=conv['question_unique_id'], captions=conv['captions'], )
    req_input = instruct_conv["conversation"][0]["human"]
    question_id = instruct_conv['conversation_id']
    pred = get_response(req_input, 2048, '<question>:\n')
    print('\n', pred)
    pred_labels.append(pred)
    answers[question_id] = pred

# 统计 pred_labels 中元素个数
answer_dict = collections.Counter(pred_labels)
print(answer_dict)
json.dump(answers, open(out_file, 'w'))
print('save to: ', out_file)
print('pred done!')
