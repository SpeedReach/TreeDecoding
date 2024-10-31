from typing import Callable, Tuple, List
from transformers import LlamaForCausalLM, LlamaTokenizer
import datasets
import time
from tqdm import tqdm


class Metric:
    def __init__(self, id: str, time_taken: float, memory_usage: List[int]):
        self.id = id
        self.time_taken = time_taken
        self.memory_usage = memory_usage

    def to_dict(self):
        return {
            "id": self.id,
            "time_taken": self.time_taken,
            "memory_usage": self.memory_usage
        }
    


def run_bench_mark(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    dataset: datasets.Dataset,
    generate: Callable[[LlamaForCausalLM, LlamaTokenizer, str, int, int], Tuple[str, List[int]]],
    num_beams = 10,
    max_tokens = 500,
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
    
    metrics_list = []
    for i in progress_bar:
        data = dataset[i]
        prompt = data['text'][:20]
        
        # Update progress bar description with current sample ID
        progress_bar.set_description(f"Processing sample {data['id']}")
        
        start = time.time()
        output, memory_usage = generate(model, tokenizer, prompt, num_beams, max_tokens)
        #for i in range(len(output)):
        #    print(":", tokenizer.decode(output[i].long()))
        end = time.time()
        
        metric = Metric(data['id'], end - start, memory_usage)
        metrics_list.append(metric)

        # Update progress bar postfix with current metrics
        progress_bar.set_postfix({
            'time': f"{metric.time_taken:.2f}s",
            'mem': f"{max(memory_usage) if memory_usage else 0:.2f}MB"
        })
    
    return metrics_list