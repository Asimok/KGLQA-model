import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

"""
使用该脚本，将lora的权重合并大base model中
"""


def merge_lora_to_base_model():
    # model_name_or_path = "/data0/maqi/huggingface_models/TechGPT-7B"
    # model_name_or_path = '/data0/maqi/huggingface_models/llama-2-7b'
    # model_name_or_path = '/data0/maqi/huggingface_models/alpaca-2-7b-english'
    # model_name_or_path = '/data0/maqi/huggingface_models/Chinese-LLaMA-Alpaca-2/Chinese-Alpaca-2-7B'
    # model_name_or_path = '/data0/maqi/huggingface_models/Chinese-LLaMA-Alpaca-2/Chinese-LLaMA-2-7b'
    # model_name_or_path = "/data0/maqi/huggingface_models/option1-models/race_ft"
    # model_name_or_path = "/data0/maqi/huggingface_models/option1-models/option1-ncr_ft"
    # model_name_or_path = "/data0/maqi/huggingface_models/baichuan/Baichuan2-7B-Base"
    model_name_or_path = '/data0/maqi/huggingface_models/chinese-llama-2-7b-64k'

    # adapter_name_or_path = '/data0/maqi/KGLQA-model/output/option-2/NCR/ncr_and_cclue_alpaca_2/final'
    adapter_name_or_path = '/data0/maqi/KGLQA-model/output/option-1/NCR/ncr_ft_alpaca_64k/final'
    # save_path = '/data0/maqi/KGLQA-model/output/RACE/race_ft'
    save_path = "/data0/maqi/huggingface_models/option1-models/option1-ncr_ft_alpaca_64k"

    print('*' * 60)
    print(f"model_name_or_path:\n {model_name_or_path}")
    print(f"adapter_name_or_path:\n {adapter_name_or_path}")
    print(f'save_path:\n {save_path}')
    print('*' * 60)

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' else True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # device_map='auto',
        device_map={'': 'cpu'}
    )
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map={'': 'cpu'})
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print('save to ', save_path)


if __name__ == '__main__':
    merge_lora_to_base_model()
