from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

"""
使用该脚本，将lora的权重合并大base model中
"""


def merge_lora_to_base_model():
    # model_name_or_path = "/data0/maqi/huggingface_models/TechGPT-7B"
    # model_name_or_path = '/data0/maqi/huggingface_models/llama-2-7b'
    # model_name_or_path = "/data0/maqi/huggingface_models/option1-models/race_ft"
    model_name_or_path = '/data0/maqi/huggingface_models/option2-models/race_ft'

    adapter_name_or_path = '/data0/maqi/KGLQA-model/output/option-2/race_1_quality_2/checkpoint-100'
    # save_path = '/data0/maqi/KGLQA-model/output/RACE/race_ft'
    save_path = '/data0/maqi/huggingface_models/option2-models/option2-quality-2048'

    config = AutoConfig.from_pretrained(model_name_or_path)
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
