import torch
import torch.nn.functional as F
from transformers import LlamaTokenizer, cache_utils, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
import time
from datasets import load_dataset
import json


from origin import origin_generate, origin_warmup
from tree_decoding import tree_generate, tree_warmup
from run import run_bench_mark, TaskType
from transformers import logging
from run import Metric
from typing import List

logging.set_verbosity_error()



model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer.pad_token_id = tokenizer.eos_token_id


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

# beams / max_tokens
parameters = [
    (3, 1000),
    (9 , 1000),
    (15 , 1000),
]





origin_warmup(model, tokenizer, "This is a test", 3, 500)

for parameter in parameters:
    out_file = open(f"out/origin/{parameter[0]}_{parameter[1]}.jsonl", "w")
    metrics = run_bench_mark(model, tokenizer, ds.select(range(1)), origin_generate, TaskType.SUM, parameter[0], parameter[1])

    for metric in metrics:
        out_file.write(json.dumps(metric.to_dict()) + "\n")


exit(0)





tree_warmup(model, tokenizer, "This is a test", 3, 500)

for parameter in parameters:
    out_file = open(f"out/tree/{parameter[0]}_{parameter[1]}.jsonl", "w")
    metrics = run_bench_mark(model, tokenizer, ds.select(range(1)), tree_generate, TaskType.SUM, parameter[0], parameter[1], max_new_tokens=1000)
    for metric in metrics:
        out_file.write(json.dumps(metric.to_dict()) + "\n")