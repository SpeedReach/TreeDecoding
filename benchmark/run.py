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


import GPUtil
import time

def get_gpu_usage():
    gpus = GPUtil.getGPUs()
    total = 0
    for gpu in gpus:
        total += gpu.memoryUsed
    return total


class Metric:

    def __init__(self, id: str,model_memory: int, time_taken: float, memory_usage: List[int], time_metric: List[float]):
        self.model_memory = model_memory
        self.input_kv_memory = memory_usage[0]
        self.id = id
        self.time_taken = time_taken
        self.memory_usage = memory_usage
        self.time_metric = time_metric

    def to_dict(self):
        return {
            "id": self.id,
            "model_memory": self.model_memory,
            "time_taken": self.time_taken,
            "input_kv_memory": self.input_kv_memory,
            "memory_usage": self.memory_usage,
            "time_metric": self.time_metric
        }
    


def run_bench_mark(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    dataset: datasets.Dataset,
    generate: Callable[[LlamaForCausalLM, LlamaTokenizer, str, int, int], Tuple[str, List[int]]],
    task_type: TaskType,
    num_beams = 10,
    max_new_tokens = 500,
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

    model_memory = get_gpu_usage()
    
    metrics_list = []
    for i in progress_bar:
        data = dataset[i]
        if task_type == TaskType.SUM:
            prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a helpful assistant capable of summarizing text accurately.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Summarize the following text:
{data['text']}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        torch.cuda.empty_cache()
        gpu_gc.collect()
        LlamaForCausalLM.clear()
        # Update progress bar description with current sample ID
        progress_bar.set_description(f"Processing sample {data['id']}")
        

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        if input_ids.shape[1] + max_new_tokens > 8000:
            break

        max_tokens = max_new_tokens + input_ids.shape[1]
        start = time.time()
        output, memory_usage, time_metric  = generate(model, tokenizer, prompt, num_beams, max_tokens)

        #if isinstance(output, str):
        #    print(":", output)
        #else:
        #    print(":", tokenizer.decode(output.long()))
        
        end = time.time()
        
        metric = Metric(
            id=data['id'],
            model_memory=model_memory,
            time_taken=end - start,
            memory_usage=memory_usage,
            time_metric=time_metric
        )
        metrics_list.append(metric)

        # Update progress bar postfix with current metrics
        progress_bar.set_postfix({
            'time': f"{metric.time_taken:.2f}s",
            'mem': f"{max(memory_usage) if memory_usage else 0:.2f}MB"
        })
    
    return metrics_list

