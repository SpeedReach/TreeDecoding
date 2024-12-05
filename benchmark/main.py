import torch
import torch.nn.functional as F
from transformers import LlamaTokenizer, cache_utils, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
import time
import datasets
from datasets import load_dataset
import json


from origin import origin_generate, origin_warmup
from tree_decoding import tree_generate, tree_warmup
from run import run_bench_mark, TaskType
from transformers import logging
from run import Metric
from typing import List

logging.set_verbosity_error()

import sys
sys.setrecursionlimit(3000)



model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer.pad_token_id = tokenizer.eos_token_id

def convert_cnn_format(d):
    return {
        'id': d['id'],
        'text': d['article'],  # 或者如果要包含摘要：example['article'] + " " + example['highlights']
        'highlights': d['highlights']
    }

def convert_human_eval_format(d):
    return {
        'id': d['id'],
        'prompt': d['prompt']
    }

def load_cnn_sum() -> datasets.Dataset:
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split='train+validation+test')
    ds = ds.map(
        convert_cnn_format,
        batched=True,
        remove_columns=['article', 'highlights']
    )
    return ds

def load_human_eval() -> datasets.Dataset:
    ds = load_dataset("openai_humaneval")
    return ds



# beams / max_tokens
parameters = [
    (1 , 1000),
    (3, 1000),
    (9 , 1000),
    (15 , 1000),
]

task_type = TaskType.HUMAN_EVAL

ds = load_human_eval() if task_type == TaskType.HUMAN_EVAL else load_cnn_sum()

tree_warmup(model, tokenizer, "This is a test", 3, 1000)

for parameter in parameters:
    out_file = open(f"out/tree/{parameter[0]}_{parameter[1]}.jsonl", "w")
    metrics = run_bench_mark(model, tokenizer, ds.select(range(1)), tree_generate, task_type, parameter[0], parameter[1])
    for metric in metrics:
        out_file.write(json.dumps(metric.to_dict()) + "\n")



origin_warmup(model, tokenizer, "This is a test", 3, 1000)

for parameter in parameters:
    out_file = open(f"out/origin/{parameter[0]}_{parameter[1]}.jsonl", "w")
    metrics = run_bench_mark(model, tokenizer, ds.select(range(1)), origin_generate, task_type, parameter[0], parameter[1])

    for metric in metrics:
        out_file.write(json.dumps(metric.to_dict()) + "\n")








