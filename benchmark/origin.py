import torch
import gc as gpu_gc
from transformers import LlamaForCausalLM
from typing import Tuple, List


def origin_warmup(model, tokenizer, prompt, num_beams, max_tokens):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)
    model.generate(input_ids, attention_mask=attention_mask, do_sample=False, num_beams=num_beams, max_new_tokens=max_tokens, temperature=None, top_p=None)

def origin_generate(model, tokenizer, prompt, num_beams, max_tokens) -> Tuple[str, List[int]]:
    torch.cuda.empty_cache()
    gpu_gc.collect()
    LlamaForCausalLM.clear_used_gpu()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    max_new_tokens = max_tokens - input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)
    outputs = model.generate(input_ids,attention_mask=attention_mask, do_sample=False, num_beams=num_beams, max_new_tokens=max_new_tokens, temperature=None, top_p = None)

    # Decode and print the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return (generated_text, LlamaForCausalLM.used_gpu)