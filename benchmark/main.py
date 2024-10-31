import torch
import torch.nn.functional as F
from transformers import LlamaTokenizer, cache_utils, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
import time
from datasets import load_dataset
import json


from origin import origin_generate
from run import run_bench_mark
from transformers import logging
logging.set_verbosity_error()

model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer.pad_token_id = tokenizer.eos_token_id

origin_out = open("out/origin.jsonl", "w")


ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split='train+validation+test')
def convert_cnn_format(d):
    return {
        'id': d['id'],
        'text': d['article']  # 或者如果要包含摘要：example['article'] + " " + example['highlights']
    }
ds = ds.map(
    convert_cnn_format,
    batched=True,
    remove_columns=['article', 'highlights']
)


metrics = run_bench_mark(model, tokenizer, ds.select(range(10)), origin_generate)
for metric in metrics:
    origin_out.write(json.dumps(metric) + "\n")
    
