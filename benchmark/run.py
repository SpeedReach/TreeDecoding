from typing import Callable, Tuple, List
from transformers import LlamaForCausalLM, LlamaTokenizer
import datasets
import time
from tqdm import tqdm
import torch
import gc as gpu_gc

from enum import Enum

class TaskType(Enum):
    SUM = 1
    HUMAN_EVAL = 2



from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex
from pynvml import nvmlDeviceGetMemoryInfo, nvmlShutdown
from functools import lru_cache

from rouge_score import rouge_scorer


import GPUtil

def get_gpu_usage():
    gpus = GPUtil.getGPUs()
    total = 0
    for gpu in gpus:
        total += gpu.memoryUsed
    return total



class Metric:

    def __init__(self, id: str,model_memory: int, time_taken: float, memory_usage: List[int], time_metric: List[float], score: float,output_len: int, output: str):
        self.model_memory = model_memory
        self.input_kv_memory = memory_usage[0]
        self.id = id
        self.time_taken = time_taken
        self.memory_usage = memory_usage
        self.time_metric = time_metric
        self.score = score
        self.output_len = output_len
        self.output = output

    def to_dict(self):
        return {
            "id": self.id,
            "model_memory": self.model_memory,
            "time_taken": self.time_taken,
            "input_kv_memory": self.input_kv_memory,
            "memory_usage": self.memory_usage,
            "time_metric": self.time_metric,
            "score": self.score,
            "output_len": self.output_len,
            "output": self.output
        }
    


def run_bench_mark(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    dataset: datasets.Dataset,
    generate: Callable[[LlamaForCausalLM, LlamaTokenizer, str, int, int], Tuple[str, List[int]]],
    task_type: TaskType,
    num_beams = 10,
    max_new_tokens = 1000,
) -> List[Metric]:
    # Create tqdm progress bar
    progress_bar = tqdm(
        range(len(dataset)),
        desc="Running benchmark",
        unit="sample",
        ncols=100,
        position=0,
        leave=True
    )

    torch.cuda.empty_cache()
    gpu_gc.collect()
    LlamaForCausalLM.clear()

    
    
    metrics_list = []
    
    for i in progress_bar:
        data = dataset[i]
        if task_type == TaskType.SUM:
            prompt = f"""<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

Output summary directly.
Article:
{data['text']}
Summary: [/INST]"""
        elif task_type == TaskType.HUMAN_EVAL:
            prompt = f"""<s>[INST] <<SYS>>
You are a programmer.
<</SYS>>
Complete the following code. No explaination is needed, output the code directly.
{problem} [/INST]"""
        torch.cuda.empty_cache()
        gpu_gc.collect()
        LlamaForCausalLM.clear()
        # Update progress bar description with current sample ID
        progress_bar.set_description(f"Processing sample {data['id']}")
        
        model_memory = get_gpu_usage()

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        if input_ids.shape[1] + max_new_tokens > 8000:
            break

        start = time.time()
        output, memory_usage, time_metric  = generate(model, tokenizer, prompt, num_beams, max_new_tokens )
        #print("shape",output.shape)
        completion = tokenizer.decode(output, skip_special_tokens=True)
        #print(":", completion)

        score = 0
        if task_type == TaskType.SUM:
            rouge = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
            score = rouge.score(completion, data['highlights'])['rouge2'].fmeasure
        
        end = time.time()
        
        metric = Metric(
            id=data['id'],
            model_memory=model_memory,
            time_taken=end - start,
            memory_usage=memory_usage,
            time_metric=time_metric,
            score=score,
            output_len=len(output),
            output=completion
        )
        metrics_list.append(metric)

        # Update progress bar postfix with current metrics
        progress_bar.set_postfix({
            'time': f"{metric.time_taken:.2f}s",
            'mem': f"{max(memory_usage) if memory_usage else 0:.2f}MB"
        })

        
    
    return metrics_list

