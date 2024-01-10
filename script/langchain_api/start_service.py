import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys

sys.path.append("../../")
from flask import Flask, request, jsonify

import torch
from transformers import AutoTokenizer, set_seed, LlamaTokenizer, GenerationConfig

from component.utils import ModelUtils

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # 防止返回中文乱码


@app.route('/techgpt-api', methods=['POST'])
def ds_llm():
    params = request.get_json()
    text = params.pop('inputs').strip()
    max_new_tokens = params.get('max_new_tokens', 500)
    top_p = params.get('top_p', 0.85)
    temperature = params.get('temperature', 0.35)
    repetition_penalty = params.get('repetition_penalty', 1.0)
    do_sample = params.get('do_sample', True)

    text = f"Human: {text} \n\nAssistant: "
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    # bos_token_id = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)
    # eos_token_id = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long).to(device)
    # input_ids = torch.concat([bos_token_id, input_ids, eos_token_id], dim=1)

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=40,
        num_beams=1,
        # bos_token_id=1,
        # eos_token_id=2,
        # pad_token_id=0,
        max_new_tokens=max_new_tokens,
        min_new_tokens=10,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty
    )
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, generation_config=generation_config
        )

    outputs = outputs.tolist()[0][len(input_ids[0]):]
    response = tokenizer.decode(outputs)
    response = response.strip().replace(tokenizer.eos_token, "").strip()
    return jsonify({"inputs": text, "response": response})


if __name__ == '__main__':
    # 使用合并后的模型进行推理
    # model_name_or_path = "/data0/maqi/huggingface_models/option1-models/cclue_ft_TechGPT-7B"
    adapter_name_or_path = None

    # 使用base model和adapter进行推理，无需手动合并权重
    model_name_or_path = "/data0/maqi/huggingface_models/TechGPT-7B"
    # model_name_or_path = "/data0/maqi/huggingface_models/firefly-llama2-7b-chat"
    # model_name_or_path = "/data0/maqi/huggingface_models/llama-2-7b"
    # model_name_or_path = "/data0/maqi/huggingface_models/option1-models/option1-ncr_ft"
    # model_name_or_path = "/data0/maqi/huggingface_models/option2-models/option2-cclue"

    # adapter_name_or_path = os.path.join('/data0/maqi/KGLQA-model/output/option-1/CCLUE/option1_ncr_1_cclue_2/final')

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

    app.run(host="0.0.0.0", port=27031)
