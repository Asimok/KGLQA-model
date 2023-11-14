import os.path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoTokenizer, LlamaTokenizer, set_seed
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


@app.route('/get_captions', methods=['POST'])
def ds_llm():
    params = request.get_json()
    text = params.pop('inputs').strip()
    max_new_tokens = params.get('max_seq_length', 2048)

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
    model_name_or_path = "/data0/maqi/huggingface_models/firefly-llama2-7b-chat"
    adapter_name_or_path = None

    set_seed(318)
    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = False
    top_p = 0.9
    temperature = 0.1
    repetition_penalty = 1.2

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
    print(f"load model: {model_name_or_path}")

    app.run(host="0.0.0.0", port=7036)
