from typing import Callable, Tuple, List
from transformers import LlamaForCausalLM, LlamaTokenizer
import datasets
import time


def run_bench_mark(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    dataset: datasets.Dataset,
    generate: Callable[[LlamaForCausalLM, LlamaTokenizer, str, int, int], Tuple[str, List[int]]],
):
    for i in range(len(dataset)):
        data = dataset[i]
        metrics = {'id': data['id'], 'time_taken': 0, 'memory_usage': []}
        prompt = data['text']
        start = time.time()
        output, memory_usage = generate(model, tokenizer, prompt, 5, 600)
        end = time.time()
        metrics['memory_usage'] = memory_usage
        metrics['time_taken'] = end - start
        print(metrics)