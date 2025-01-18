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
from run import run_bench_mark, TaskType, ModelType
from transformers import logging
from run import Metric
from typing import List
import os

logging.set_verbosity_error()

import sys
sys.setrecursionlimit(3000)

model_type = ModelType.PHI35

if model_type == ModelType.LLAMA2:
    model_name = "meta-llama/Llama-2-7b-chat-hf" 
elif model_type == ModelType.PHI35:
    model_name = "microsoft/Phi-3.5-mini-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)


def convert_cnn_format(d):
    return {
        'id': d['id'],
        'text': d['article'],  # 或者如果要包含摘要：example['article'] + " " + example['highlights']
        'answer': d['highlights']
    }

def convert_human_eval_format(d):
    return {
        'id': d['task_id'],
        'text': d['prompt']
    }

def convert_qasper_format(d):
    full_text = json.loads(d['full_text'])
    doc = ""
    for paragraph in full_text["paragraphs"]:
        doc += paragraph
        doc += "\n"
    qas = json.loads(d['qas'])
    doc = f"""
    Answer the question according to the following document, or answer "I don't know" if the document doesn't contain the answer.
    Doc:
    {doc}
    Question:
    {qas["question"][0]}
    """
    answer = ""
    for ans in qas["answers"][0]["answer"]:
        answer = ans["free_form_answer"]
        if answer != "":
            break
    if answer == "":
        answer = "I don't know."
    return {
        'id': d['id'],
        'text': doc,
        'answer': answer
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
    ds = load_dataset("openai_humaneval", split='test')
    ds = ds.map(
        convert_human_eval_format,
        batched=True
    )
    return ds

def load_qasper() -> datasets.Dataset:
    ds = load_dataset("allenai/qasper", split='train')
    ds = ds.map(convert_qasper_format, batched=True)
    return ds

# beams / max_tokens
parameters = [
#    (1 , 1000),
    (3, 1000),
#    (9 , 1000),
#    (15 , 1000),
]



def run_task(task_type: TaskType, data_num: int):
    match task_type:
        case TaskType.HUMAN_EVAL:
            ds = load_human_eval()
        case TaskType.SUM:
            ds = load_cnn_sum()
        case TaskType.QASPER:
            ds = load_qasper()

    tree_warmup(model, tokenizer, "This is a test", 3, 1000,  [ model.config.eos_token_id ])

    for parameter in parameters:
        if parameter[0] == 1:
            continue

        path = f"out/tree/{task_type.name}"
        os.makedirs(path, exist_ok=True)
        print("processing tree ",parameter[0], "_",parameter[1] )
        with open(f"{path}/{parameter[0]}_{parameter[1]}.jsonl", "w") as out_file:
            metrics = run_bench_mark(model, tokenizer, ds.select(range(data_num)), tree_generate, task_type, model_type, parameter[0], parameter[1])
            for metric in metrics:
                out_file.write(json.dumps(metric.to_dict()) + "\n")

    origin_warmup(model, tokenizer, "This is a test", 3, 1000)

    for parameter in parameters:
        path = f"out/origin/{task_type.name}"
        os.makedirs(path, exist_ok=True)
        print("processing origin ",parameter[0], "_",parameter[1] )
        with open(f"{path}/{parameter[0]}_{parameter[1]}.jsonl", "w") as out_file:
            metrics = run_bench_mark(model, tokenizer, ds.select(range(data_num)), origin_generate, task_type, model_type, parameter[0], parameter[1])
            for metric in metrics:
                out_file.write(json.dumps(metric.to_dict()) + "\n")



run_task(TaskType.QASPER, 1)

#run_task(TaskType.HUMAN_EVAL, 1)
