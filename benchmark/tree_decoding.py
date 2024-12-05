import torch
import gc as gpu_gc
from transformers import LlamaForCausalLM
from typing import Tuple, List, Optional
from torch import Tensor
from transformers.cache_utils import DynamicCache

from typing import List, Tuple
import time

class SearchNode:
    def __init__(self, root, idx, token_id, token_score):
        self.root: 'SearchTree' = root
        self.idx: int = idx
        self.token_id: Tensor = token_id
        self.token_score: torch.FloatTensor = token_score
        self.parent: Optional['SearchNode'] = None
        self.children: List['SearchNode'] = []
        self.acc_score: torch.FloatTensor = token_score


    def add_children(self, child):
        self.children.append(child)
        child.parent = self
        self.root.node_count += 1

    def delete_child(self, child):
        self.children.remove(child)
        self.root.node_count -= 1


class SearchTree:
    def __init__(self, model, beam_width=3):
        self.node_count: int = 0
        self.model = model
        self.device = model.device
        self.root: List[SearchNode] = []
        self.beam_width: int = beam_width



def dfs(searchNode: SearchNode, targets: List[int], traversed: List[int]) -> Tuple[bool, List[int], List[int]]:
    # returns found, found path, unused nodes
    traversed.append(searchNode.idx)
    if searchNode.idx in targets:
        return (True, traversed, [])
    
    if len(searchNode.children) == 0:
        return (False, [], traversed)
    
    if len(searchNode.children) == 1:
        return dfs(searchNode.children[0], targets, traversed)
    
    child_found = False
    found_path = []
    unused = []
    for child in searchNode.children:
        found, fp, u = dfs(child, targets, [])
        if found:
            found_path += fp
            child_found = True
        unused += u

    if child_found:
        found_path = traversed + found_path
    else:
        unused = traversed + unused
    
    return (child_found, found_path, unused)


def determine_unused_nodes(searchTree: SearchTree, targets: List[int]) -> Tuple[List[int], List[int]]:
    all_unused = []
    all_used = []
    for child in searchTree.root:
        _, used, unused = dfs(child, targets, [])
        all_unused += unused
        all_used += used
    return (all_used, all_unused)


minFloat = torch.finfo(torch.float16).min

def fill_causal_mask(mask: torch.Tensor, searchTree: SearchTree,input_len: int,nodes: List[SearchNode]):
    mask.fill_(minFloat)
    branch_count = len(nodes)
    mask[0, 0,:,:input_len] = 0
    tmp = nodes.copy()
    while True:
        end = False
        for i in range(branch_count):
            #print(i, tmp[i].idx)
            mask[0, 0, i, tmp[i].idx + input_len] = 0
            if tmp[i].parent is not None:
                tmp[i] = tmp[i].parent
            else:
                end = True
        if end:
            return

def fill_causal_mask_fast(mask: torch.Tensor, searchTree: 'SearchTree', input_len: int, nodes: List['SearchNode']):
    mask.fill_(minFloat)  # Initialize all values in the mask
    branch_count = len(nodes)
    mask[0, 0, :, :input_len] = 0  # Set the initial mask for input length

    indices = []  # Store indices where we need to set zeroes
    tmp = nodes.copy()

    while True:
        end = True  # Assume this will be the last iteration unless we find a parent
        for i in range(branch_count):
            indices.append((0, 0, i, tmp[i].idx + input_len))  # Collect indices for later use
            if tmp[i].parent is not None:
                tmp[i] = tmp[i].parent
                end = False  # If any node has a parent, continue the loop
        if end:
            break

    # Convert indices to a tensor and set values to zero in one GPU operation
    indices_tensor = torch.tensor(indices, dtype=torch.long, device=mask.device).T
    mask[indices_tensor[0], indices_tensor[1], indices_tensor[2], indices_tensor[3]] = 0

def fill_mask_end(mask: torch.Tensor, searchTree: SearchTree,input_len: int,nodes: List[SearchNode]):
    for i in range(len(nodes)):
        mask[0,0, i, nodes[i].idx + input_len] = 0
    return


def print_tree_state(searchTree: SearchTree,nodes: List[SearchNode]):
    branch_count = len(nodes)
    tmp = nodes.copy()
    print("========")
    print("node count: ", searchTree.node_count)
    while True:
        end = False
        for i in range(branch_count):
            print(i, tmp[i].idx)
            if tmp[i].parent is not None:
                tmp[i] = tmp[i].parent
            else:
                end = True
        if end:
            return


import torch
import torch.nn.functional as F
from collections import deque


def prune_kv_cache(past_key_values, input_length, remove_idx: List[int]):
    device = past_key_values[0][0].device
    remove_idx = [i + input_length for i in remove_idx]
    #print("remove", remove_idx)
    all_indices = torch.arange(past_key_values[0][0].size(2), device = device)

    keep_indices = all_indices[~torch.isin(all_indices, torch.tensor(remove_idx, device=device))]
    #print("keep", keep_indices)

    for i in range(len(past_key_values)):
        if keep_indices.device != past_key_values.key_cache[i].device:
            keep_indices= keep_indices.to(past_key_values.key_cache[i].device)
        past_key_values.key_cache[i] = torch.index_select(past_key_values.key_cache[i], 2, keep_indices)
        past_key_values.value_cache[i] = torch.index_select(past_key_values.value_cache[i], 2, keep_indices)

def prune_tree(searchTree: SearchTree, remove_idx: List[int]):
    for child in searchTree.root[:]:
        if child.idx in remove_idx:
            #print("removed ", child.idx)
            searchTree.root.remove(child)
    tmp = deque(searchTree.root)
    while len(tmp) > 0:
        node = tmp.popleft()
        for child in node.children[:]:
            if child.idx in remove_idx:
                #print("removed ", child.idx)
                node.children.remove(child)
            else:
                tmp.append(child)
    i = 0

    tmp = deque(searchTree.root)
    while len(tmp) > 0:
        children = []
        while len(tmp) > 0:
            node = tmp.popleft()
            node.idx = i
            i += 1
            for child in node.children:
                children.append(child)
        if len(children) > 1:
            children = sorted(children, key=lambda node: node.idx)
        tmp.extend(children)
    searchTree.node_count = i

def gc(searchTree: SearchTree,input_length, newest_branch: List[SearchNode], past_key_values):
    find_unused_start = time.time()
    unused = determine_unused_nodes(searchTree, [ node.idx for node in newest_branch])
    #print("find_unused takes", time.time() - find_unused_start)
    #print("Unused: ", len(unused[1]), len(unused[0]) + len(unused[1]) , unused)
    prune_tree_start = time.time()
    prune_tree(searchTree, unused[1])
    #print("prune tree takes ",time.time()-prune_tree_start)

    prune_kv_start = time.time()
    kv = prune_kv_cache(past_key_values,input_length, unused[1])
    #print("prune kv takes", time.time() - prune_kv_start)
    #print_tree_state(searchTree, newest_branch)
    return 

def is_done(self, best_sum_logprobs: float, cur_len: int, decoder_prompt_len: Optional[int] = 0) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False

        # `True`: stop as soon as at least `num_beams` hypotheses are finished
        if self.early_stopping is True:
            return True
        # `False`: heuristic -- compute best possible score from `cur_len`, even though it is not entirely accurate
        #  when `length_penalty` is positive. See the discussion below for more details.
        # https://github.com/huggingface/transformers/pull/20901#issuecomment-1369845565
        elif self.early_stopping is False:
            highest_attainable_score = best_sum_logprobs / (cur_len - decoder_prompt_len) ** self.length_penalty
            ret = self.worst_score >= highest_attainable_score
            return ret
        # `"never"`: compute the best possible score, depending on the signal of `length_penalty`
        else:
            # `length_penalty` > 0.0 -> max denominator is obtaned from `max_length`, not from `cur_len` -> min
            # abs(`highest_attainable_score`) is obtained -> `highest_attainable_score` is negative, hence we obtain
            # its max this way
            if self.length_penalty > 0.0:
                if self.max_length <= decoder_prompt_len:
                    raise ValueError("max_length is not larger than decoder prompt length")
                highest_attainable_score = (
                    best_sum_logprobs / (self.max_length - decoder_prompt_len) ** self.length_penalty
                )
            # the opposite logic applies here (max `highest_attainable_score` from `cur_len`)
            else:
                highest_attainable_score = best_sum_logprobs / (cur_len - decoder_prompt_len) ** self.length_penalty
            ret = self.worst_score >= highest_attainable_score
            return ret

@torch.no_grad()
def generate_next_tokens(model, input_ids, beam_width = 3, max_new_tokens=300) -> Tuple[torch.Tensor, List[int], List[float]]:
    past_key_values = DynamicCache()
    input_len = input_ids.shape[1]
    #print("input length: ", input_len)

    device = model.device

    #generate the first 3 tokens
    outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values
    token_scores = F.log_softmax(outputs.logits, dim=-1)

    token_scores, tokens = torch.topk(token_scores, beam_width, dim=-1, largest=True, sorted=True)
    #print("picks ", tokens[0][-1])
    searchTree = SearchTree(model = model, beam_width = beam_width)
    newest_branch: List[SearchNode] = []
    idx = 0

    #define eos token
    eos_token_id = model.config.eos_token_id

    
    for i in range(beam_width):
        searchNode = SearchNode(searchTree, idx, tokens[0][-1][i], token_scores[0][-1][i])
        idx += 1
        newest_branch.append(searchNode)
        searchTree.root.append(searchNode)
        searchTree.node_count += 1
    
    completed_branches = []
    alive_beams = beam_width

    need_gc = False
    attention_mask = torch.full((1, 1, beam_width, max_new_tokens * beam_width+input_len), minFloat, device=device, dtype=torch.float16)
    fill_causal_mask(attention_mask, searchTree, input_len, newest_branch)
    next_indices = [x for x in range(beam_width) ] 
    early_complete = False
    for i in range(input_len, max_new_tokens+input_len):
        if  ((i % 30 == 0 and alive_beams > 2) or need_gc) and True:
            gc_start = time.time()
            #print("gcccc")
            need_gc = False
            gc(searchTree,input_len, newest_branch, past_key_values)
            idx = searchTree.node_count
            fill_mask_start = time.time()
            fill_causal_mask_fast(attention_mask, searchTree, input_len, newest_branch)
            if attention_mask.shape[2] != alive_beams:
                attention_mask = attention_mask[:,:,:alive_beams,:]
            #print("fill mask takes ", time.time()-fill_mask_start)
            gc_end = time.time()
            #print(f"gc took {gc_end-gc_start}")
        elif i != input_len:
            #fill_causal_mask(attention_mask, searchTree, input_len, newest_branch)
            reorder_mask_start = time.time()
            result = attention_mask.clone()
            for x in range(alive_beams):
                result[:, :, x, :] = attention_mask[:, :, next_indices[x], :]
            attention_mask = result
            fill_mask_end(attention_mask, searchTree,input_len,newest_branch.copy())
            #print("reorder mask takes ",time.time() -  reorder_mask_start)
        #construct position_ids
        #print("alive_beams: ", alive_beams)
        position_ids = torch.tensor([[i for _ in range(alive_beams)]], device=model.device)

        #construct input_ids
        input_ids = torch.tensor([[node.token_id for node in newest_branch]], device=model.device)
        
        #generate candidate tokens
        pass_start = time.time()
        outputs = model(input_ids, past_key_values=past_key_values, position_ids=position_ids, attention_mask=attention_mask, use_cache=True)
        pass_end = time.time()
        #print("One pass takes", pass_end - pass_start)
        past_key_values = outputs.past_key_values
        #calculate token scores
        token_scores = F.log_softmax(outputs.logits, dim=-1)

        beam_score = torch.tensor([b.acc_score for b in newest_branch], device=model.device)
        beam_score = beam_score.view((1, 1, alive_beams, 1))
        token_scores = token_scores + beam_score
        
        vocab_size = token_scores.shape[-1]
        token_scores = token_scores.view(alive_beams * vocab_size)
        n_eos_tokens = len(eos_token_id) if eos_token_id is not None else 0
        n_tokens_to_keep = max(2, 1 + n_eos_tokens) * alive_beams
        token_scores, tokens = torch.topk(
            token_scores, n_tokens_to_keep, dim=0, largest=True, sorted=True
        )
        #which parent
        next_indices = torch.div(tokens, vocab_size, rounding_mode="floor") 
        #tokens
        tokens = tokens % vocab_size

        #update newest_branch and searchTree

        tmp_newest_branch = []
        
        completed_nodes = []
        picked = []
        picked_scores = []
        final_picked_parents = []
        
        for j in range(len(tokens)):
            token_id = tokens[j]
            picked.append(token_id.item())
            searchNode = SearchNode(searchTree, idx, token_id=token_id, token_score = token_scores[j])
            
            #print(int(token_idx/beam_width)," add child")
            
            if token_id in eos_token_id:
                #print(i, "ended")
                #need_gc = True
                completed_nodes.append(searchNode)
                completed_branches.append(searchNode)
                searchNode.parent = newest_branch[next_indices[j]]
                #tmp_newest_branch.append(searchNode)
            else:
                picked_scores.append(token_scores[j].item())
                newest_branch[next_indices[j]].add_children(searchNode)
                final_picked_parents.append(next_indices[j]) #- len(completed_nodes))
                idx += 1
                tmp_newest_branch.append(searchNode)

            if len(tmp_newest_branch) >= alive_beams:
                break
        #print(i, picked_scores)
        next_indices = final_picked_parents
        #print("picks ", picked)
        #print("picked_scores ", picked_scores)
        #alive_beams -= len(completed_nodes)
        newest_branch = tmp_newest_branch
        #for metrics we remove early stop
        if len(completed_branches) >= beam_width:
            early_complete = True
            break

    #construct the output
    outputs = []
    if early_complete:
        newest_branch = completed_branches
    else:
        newest_branch = newest_branch + completed_branches
    for i in range(len(newest_branch)):
        output = torch.empty(0, device=model.device)
        branch_parent = newest_branch[i]
        length = 0
        score = branch_parent.acc_score
        while branch_parent is not None:
            length += 1
            output = torch.cat((output, branch_parent.token_id.unsqueeze(0)))
            branch_parent = branch_parent.parent
        output=output.flip(dims=[0])
        outputs.append((output, score / length))
        #outputs = torch.cat((outputs, output.unsqueeze(0)))
    max_score = max(x[1] for x in outputs)
    max_sequence = [x[0] for x in outputs if x[1] == max_score]
    return (max_sequence[0], LlamaForCausalLM.used_gpu, LlamaForCausalLM.time_metric)



def tree_warmup(model, tokenizer, prompt, num_beams, max_tokens):
    tree_generate(model, tokenizer, prompt, num_beams, max_tokens)

def tree_generate(model, tokenizer, prompt, num_beams, max_new_tokens) -> Tuple[str, List[int], List[float]]:
    torch.cuda.empty_cache()
    gpu_gc.collect()
    LlamaForCausalLM.clear()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output = generate_next_tokens(model, input_ids, beam_width=num_beams, max_new_tokens=max_new_tokens)
    return (tokenizer.decode(output[0].long(), skip_special_tokens=True), output[1], output[2])


