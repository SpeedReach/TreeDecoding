from typing import Callable, Tuple, List
from transformers import LlamaForCausalLM, LlamaTokenizer
import datasets
import time
from tqdm import tqdm

def run_bench_mark(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    dataset: datasets.Dataset,
    generate: Callable[[LlamaForCausalLM, LlamaTokenizer, str, int, int], Tuple[str, List[int]]],
):
    # Create tqdm progress bar
    progress_bar = tqdm(
        range(len(dataset)),
        desc="Running benchmark",
        unit="sample",
        ncols=100,
        position=0,
        leave=True
    )
    
    metrics_list = []
    for i in progress_bar:
        data = dataset[i]
        metrics = {'id': data['id'], 'time_taken': 0, 'memory_usage': []}
        prompt = data['text']
        
        # Update progress bar description with current sample ID
        progress_bar.set_description(f"Processing sample {data['id']}")
        
        start = time.time()
        output, memory_usage = generate(model, tokenizer, prompt, 5, 600)
        end = time.time()
        
        metrics['memory_usage'] = memory_usage
        metrics['time_taken'] = end - start
        metrics_list.append(metrics)
        print(metrics)
        
        # Update progress bar postfix with current metrics
        progress_bar.set_postfix({
            'time': f"{metrics['time_taken']:.2f}s",
            'mem': f"{max(memory_usage) if memory_usage else 0:.2f}MB"
        })
    
    return metrics_list