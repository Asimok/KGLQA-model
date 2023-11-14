import collections
import sys

sys.path.append('../../')
import json

from tqdm import tqdm

from script.evaluate.evaluate import get_response

LABEL_TO_ID_DICT = {"A": 1, "B": 2, "C": 3, "D": 4}


def instruct_format(context, answer, query, options, conversation_id):
    passage, answer, q, options = context, answer, query, options
    options = [option[2:] for option in options]

    prefix = ('Read the following passage and questions, then choose the right answer from options, the answer '
              'should be one of A, B, C, D.\n\n')
    passage = f'<passage>:\n{passage}\n\n'
    question = f'<question>:\n{q}\n\n'
    option = f'<options>:\nA {options[0]}\nB {options[1]}\nC {options[2]}\nD {options[3]}\n\n'
    suffix = f"<answer>:\n"
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


test_file = '/data0/maqi/KGLQA-data/datasets/QuALITY/quality_rocketqa_2048/test.jsonl'
with open(test_file, 'r') as f:
    samples = f.readlines()
pred_labels = []
answers = []
correct = 0
for i, human_input in enumerate(tqdm(samples)):
    conv = json.loads(human_input.strip())
    instruct_conv = instruct_format(context=conv['context'], answer='-1', query=conv['query'],
                                    options=[conv['option_0'], conv['option-1'], conv['race'],
                                             conv['option_3']], conversation_id=conv['question_unique_id'])
    req_input = instruct_conv["conversation"][0]["human"]
    question_id = instruct_conv['conversation_id']
    pred = get_response(req_input, 2048, '<question>:\n')
    pred_labels.append(pred)
    answers.append(','.join([question_id, str(LABEL_TO_ID_DICT[pred])]))
with open('/data0/maqi/KGLQA-data/datasets/QuALITY/predictions/pred.txt', 'w') as f:
    for line in answers:
        f.write(line)
        f.write('\n')
# 统计 pred_labels 中元素个数
answer_dict = collections.Counter(pred_labels)
print(answer_dict)
print('save to: /data0/maqi/KGLQA-data/datasets/QuALITY/predictions/pred.txt')
print('pred done!')
