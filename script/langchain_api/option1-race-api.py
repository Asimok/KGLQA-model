import os.path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import LlamaTokenizer, set_seed
import torch
from flask import Flask, request, jsonify

import sys

sys.path.append("../../")
from component.utils import ModelUtils

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # 防止返回中文乱码


@app.route('/option1_race_api', methods=['POST'])
def ds_llm():
    params = request.get_json()
    text = params.pop('inputs').strip()
    max_seq_length = params.get('max_seq_length', 2048)
    top_p = params.get('top_p', 0.99)
    temperature = params.get('temperature', 0.01)
    split_token = params.get('split_token', '<question>:\n')
    max_new_tokens = params.get('max_new_tokens', 1)
    repetition_penalty = params.get('repetition_penalty', 1.2)

    query = text.split(split_token)
    if len(query) == 2:
        query_1_len = len(tokenizer.encode(query[1]))
        need_token = max_seq_length - query_1_len - 2 - 4
        query[0] = tokenizer.decode(tokenizer.encode(query[0])[:need_token], skip_special_tokens=True)
        text = query[0] + split_token + query[1]

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
    model_name_or_path = "/data0/maqi/huggingface_models/option1-models/option1-race_ft"
    adapter_name_or_path = None

    print(f"model_name_or_path:\n {model_name_or_path}")
    print(f"adapter_name_or_path:\n {adapter_name_or_path}")
    set_seed(318)
    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = False

    device = 'cuda'
    # 加载模型
    model = ModelUtils.load_model(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        adapter_name_or_path=adapter_name_or_path
    ).eval()
    # 加载tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"

    print(f"load model: {model_name_or_path}")

    app.run(host="0.0.0.0", port=27034)
