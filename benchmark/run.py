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



from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex
from pynvml import nvmlDeviceGetMemoryInfo, nvmlShutdown
from functools import lru_cache

from rouge_score import rouge_scorer

class GPUMemoryMonitor:
    def __init__(self):
        """Initialize NVML"""
        nvmlInit()
        self._device_count = nvmlDeviceGetCount()
        self._device_handles = [nvmlDeviceGetHandleByIndex(i) for i in range(self._device_count)]
        
    @lru_cache(maxsize=1)
    def get_gpu_memory(self):
        """
        Get GPU memory usage using NVML
        Returns: Dictionary with GPU index and memory usage in MB
        """
        try:
            gpu_memory = {}
            for idx, handle in enumerate(self._device_handles):
                info = nvmlDeviceGetMemoryInfo(handle)
                gpu_memory[f'gpu_{idx}'] = {
                    'used': info.used // 1024 // 1024,  # Convert to MB
                    'total': info.total // 1024 // 1024,
                    'free': info.free // 1024 // 1024
                }
            return gpu_memory
        except Exception as e:
            return f"Error getting GPU info: {str(e)}"
    
    def __del__(self):
        """Cleanup NVML"""
        try:
            nvmlShutdown()
        except:
            pass

monitor = GPUMemoryMonitor()
def get_gpu_usage():
    memory_info = monitor.get_gpu_memory()
    total = 0
    for k in memory_info:
        info = memory_info[k]
        total += info['used']
    return total



class Metric:

    def __init__(self, id: str,model_memory: int, time_taken: float, memory_usage: List[int], time_metric: List[float], score: float):
        self.model_memory = model_memory
        self.input_kv_memory = memory_usage[0]
        self.id = id
        self.time_taken = time_taken
        self.memory_usage = memory_usage
        self.time_metric = time_metric
        self.score = score

    def to_dict(self):
        return {
            "id": self.id,
            "model_memory": self.model_memory,
            "time_taken": self.time_taken,
            "input_kv_memory": self.input_kv_memory,
            "memory_usage": self.memory_usage,
            "time_metric": self.time_metric,
            "score": self.score
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

    model_memory = get_gpu_usage()
    
    metrics_list = []
    rouge=rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    for i in progress_bar:
        data = dataset[i]
        if task_type == TaskType.SUM:
            prompt = f"""
Article:
{data['text']}
Summary:
            """
        torch.cuda.empty_cache()
        gpu_gc.collect()
        LlamaForCausalLM.clear()
        # Update progress bar description with current sample ID
        progress_bar.set_description(f"Processing sample {data['id']}")
        

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        if input_ids.shape[1] + max_new_tokens > 8000:
            break

        start = time.time()
        output, memory_usage, time_metric  = generate(model, tokenizer, prompt, num_beams, max_new_tokens )

        print(":", output)

        rouge_score = rouge.score(output, data['highlights'])
        print("rouge_score", rouge_score)
        
        end = time.time()
        
        metric = Metric(
            id=data['id'],
            model_memory=model_memory,
            time_taken=end - start,
            memory_usage=memory_usage,
            time_metric=time_metric,
            score=rouge_score
        )
        metrics_list.append(metric)

        # Update progress bar postfix with current metrics
        progress_bar.set_postfix({
            'time': f"{metric.time_taken:.2f}s",
            'mem': f"{max(memory_usage) if memory_usage else 0:.2f}MB"
        })

        
    
    return metrics_list

