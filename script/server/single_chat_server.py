import os.path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import AutoTokenizer, LlamaTokenizer
import torch
from flask import Flask, request, jsonify

import sys

sys.path.append("../../")
from component.utils import ModelUtils

"""
单轮对话，不具有对话历史的记忆功能
"""

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # 防止返回中文乱码


@app.route('/firefly', methods=['POST'])
def ds_llm():
    params = request.get_json()
    text = params.pop('inputs').strip()
    max_seq_length = params.get('max_seq_length', 1536)
    split_token = params.get('split_token', '<question>:\n')

    query = text.split(split_token)
    if len(query) == 2:
        # print(f"query: {len(query)}")
        query_1_len = len(tokenizer.encode(query[1]))
        need_token = max_seq_length - query_1_len - 2 - 4
        query[0] = tokenizer.decode(tokenizer.encode(query[0])[:need_token], skip_special_tokens=True)
        text = query[0] + split_token + query[1]

    # chatglm使用官方的数据组织格式
    if model.config.model_type == 'chatglm':
        text = '[Round 1]\n\n问：{}\n\n答：'.format(text)
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    # 为了兼容qwen-7b，因为其对eos_token进行tokenize，无法得到对应的eos_token_id
    else:
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        bos_token_id = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)
        eos_token_id = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long).to(device)
        input_ids = torch.concat([bos_token_id, input_ids, eos_token_id], dim=1)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
            top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id
        )
    outputs = outputs.tolist()[0][len(input_ids[0]):]
    response = tokenizer.decode(outputs)
    response = response.strip().replace(tokenizer.eos_token, "").strip()
    return jsonify({"inputs": text, "response": response})


if __name__ == '__main__':
    # 使用合并后的模型进行推理
    # model_name_or_path = '/home/jclian91/experiment/Firefly/script/checkpoint/firefly-llama2-7b-qlora-sft-race-merge'
    # adapter_name_or_path = None

    # 使用base model和adapter进行推理，无需手动合并权重
    model_name_or_path = "/data0/maqi/huggingface_models/TechGPT-7B"
    # model_name_or_path = "/data0/maqi/huggingface_models/llama-2-7b"
    # model_name_or_path = "/data0/maqi/huggingface_models/option1-models/race_ft"

    adapter_name_or_path = os.path.join(
        '/data0/maqi/KGLQA-model/output/CCLUE/cclue_ft/checkpoint-400')
    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = False
    # 生成超参配置
    max_new_tokens = 1
    top_p = 0.99
    temperature = 0.01
    repetition_penalty = 1

    device = 'cuda'
    # 加载模型
    model = ModelUtils.load_model(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        adapter_name_or_path=adapter_name_or_path
    ).eval()
    # 加载tokenizer
    if model.name_or_path.__contains__('TechGPT-7B'):
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.padding_side = "left"
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            # llama不支持fast
            use_fast=False if model.config.model_type == 'llama' else True
        )

    # QWenTokenizer比较特殊，pad_token_id、bos_token_id、eos_token_id均为None。eod_id对应的token为<|endoftext|>
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    print(f"load model: {model_name_or_path}")

    app.run(host="0.0.0.0", port=7032)
