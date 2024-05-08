#date: 2024-05-08T16:49:58Z
#url: https://api.github.com/gists/ce1adfad1d561c1e8dc92666ab5a9e8c
#owner: https://api.github.com/users/Cyrilvallez


# BENCHMARK 1
# INITIAL BENCHMARK WHEN OPENING THE PR


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from tqdm import tqdm
import numpy as np
import json
import gc

# Random generated text
LARGE_TEXT = """Title: Monkeys: Nature's Pranksters, Social Geniuses, and Ecological Wonders

Introduction

Monkeys, the charismatic and diverse members of the primate order, have long held a special place in the annals of our fascination with the animal kingdom. With their playful antics, astonishing intelligence, and complex social structures, they serve as a source of both joy and profound scientific inquiry. In this comprehensive exploration, we embark on a deep dive into the world of monkeys, spanning their evolutionary history, classifications, ecological roles, social dynamics, communication methods, and the pressing need for conservation. These captivating creatures offer insights into the intricacies of the natural world, our own evolutionary heritage, and the urgent importance of preserving biodiversity.

I. Evolutionary Origins

To understand the world of monkeys, we must embark on a journey through their evolutionary past, a tapestry that stretches back millions of years. Monkeys are part of the grand order of Primates, and their lineage is interwoven with the broader history of these remarkable mammals.

A. Primate Origins

The story of primates, including monkeys, begins around 60 million years ago. At that time, the world was a vastly different place, dominated by the reign of dinosaurs. It was during this period of Earth's history that the first primates, known as prosimians, emerged. These small, tree-dwelling mammals exhibited several characteristics that would become hallmarks of all primates: grasping hands and feet, forward-facing eyes for stereoscopic vision, and an enlarged brain relative to body size. These adaptations suited them for life in the trees, where they foraged for insects and fruits.

B. The Emergence of Monkeys

Around 35 million years ago, a significant split occurred within the primate family tree, leading to the emergence of two major groups: New World monkeys (Platyrrhini) and Old World monkeys (Catarrhini). This evolutionary divergence set in motion a cascade of adaptations that would result in the striking diversity of monkeys we see today.

The division between New World and Old World monkeys was not merely a matter of geographical separation but also marked significant differences in physical traits and behaviors. New World monkeys, found in Central and South America, are characterized by their prehensile tails and a variety of adaptations that allow them to thrive in the lush forests of the Americas. Old World monkeys, on the other hand, are residents of Africa, Asia, and parts of Gibraltar, and they have developed their own unique set of features to suit the diverse environments they inhabit.

II. Classification and Diversity"""

N_repeat = 5

torch.manual_seed(1)


def main(model_name, dtype, restrict_context_size, use_fast):

    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='flash_attention_2',
                                            torch_dtype=dtype, low_cpu_mem_usage=True).cuda(1)
    tokenizer = "**********"=use_fast)
    inputs = "**********"='pt').to(device=1)
    # Useless in our case but silences warning
    eos_token_id = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"h "**********"a "**********"s "**********"a "**********"t "**********"t "**********"r "**********"( "**********"e "**********"o "**********"s "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********", "**********"  "**********"' "**********"_ "**********"_ "**********"l "**********"e "**********"n "**********"_ "**********"_ "**********"' "**********") "**********"  "**********"a "**********"n "**********"d "**********"  "**********"h "**********"a "**********"s "**********"a "**********"t "**********"t "**********"r "**********"( "**********"e "**********"o "**********"s "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********", "**********"  "**********"' "**********"_ "**********"_ "**********"g "**********"e "**********"t "**********"i "**********"t "**********"e "**********"m "**********"_ "**********"_ "**********"' "**********") "**********": "**********"
        eos_token_id = "**********"

    if restrict_context_size:
        N_toks = [500, 1000, 2000, 3000, 3500]
    else:
        N_toks = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
    results_time = {}
    results_memory = {}
    filename = 'patched_' + model_name.split('/', 1)[1] + '_batch_size_1_input_size_300'
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"N "**********"_ "**********"t "**********"o "**********"k "**********"  "**********"i "**********"n "**********"  "**********"t "**********"q "**********"d "**********"m "**********"( "**********"N "**********"_ "**********"t "**********"o "**********"k "**********"s "**********", "**********"  "**********"d "**********"e "**********"s "**********"c "**********"= "**********"' "**********"N "**********"e "**********"w "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"' "**********") "**********": "**********"

        times = []
        memory = []
        for i in tqdm(range(N_repeat), leave=False):
            torch.cuda.reset_peak_memory_stats(1)
            actual_peak = torch.cuda.max_memory_allocated(1) / 1024**3
            t0 = time.time()
            out = model.generate(inputs[: "**********":300] "**********"max_new_tokens=N_tok "**********"min_new_tokens=N_tok "**********"do_sample=True,
                                 num_return_sequences=1, temperature=0.8, top_k=50, top_p=0.9,
                                 pad_token_id= "**********"
            dt = time.time() - t0
            memory_used = (torch.cuda.max_memory_allocated(1) / 1024**3) - actual_peak
            # Verify that we did generate the correct number of tokens
            assert out.shape[-1] == 300 + N_tok
            times.append(dt)
            memory.append(memory_used)

        results_time[N_tok] = np.mean(times)
        results_memory[N_tok] = np.mean(memory)
        # print(f'New tokens: "**********":.3e} s --- {results_memory[N_tok]:.2f} GiB')
        
        # Rewrite results at each iteration (so that we keep them if OOM)
        with open(filename + '_memory.json', 'w') as fp:
            json.dump(results_memory, fp, indent='\t')
        with open(filename + '_time.json', 'w') as fp:
            json.dump(results_time, fp, indent='\t')


    batch_sizes = [2, 4, 6, 8, 10]
    results_time = {}
    results_memory = {}
    filename = "**********"
    for batch_size in tqdm(batch_sizes, desc='Batch sizes'):

        times = []
        memory = []
        for i in tqdm(range(N_repeat), leave=False):
            torch.cuda.reset_peak_memory_stats(1)
            actual_peak = torch.cuda.max_memory_allocated(1) / 1024**3
            t0 = time.time()
            out = model.generate(inputs[: "**********":300] "**********"max_new_tokens=2000 "**********"min_new_tokens=2000 "**********"do_sample=True,
                                 num_return_sequences=batch_size, temperature=0.8, top_k=50, top_p=0.9,
                                 pad_token_id= "**********"
            dt = time.time() - t0
            memory_used = (torch.cuda.max_memory_allocated(1) / 1024**3) - actual_peak
            # Verify that we did generate the correct number of tokens
            assert out.shape[-1] == 2300
            assert out.shape[0] == batch_size
            times.append(dt)
            memory.append(memory_used)

        results_time[batch_size] = np.mean(times)
        results_memory[batch_size] = np.mean(memory)
        # print(f'Batch size: {batch_size} --- {results_time[N_tok]:.3e} s --- {results_memory[N_tok]:.2f} GiB')
        
        # Rewrite results at each iteration (so that we keep them if OOM)
        with open(filename + '_memory.json', 'w') as fp:
            json.dump(results_memory, fp, indent='\t')
        with open(filename + '_time.json', 'w') as fp:
            json.dump(results_time, fp, indent='\t')


    del model
    gc.collect()



if __name__ == '__main__':

    model_names = ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-2-7b-hf', 'meta-llama/Meta-Llama-3-8B']
    dtypes = [torch.bfloat16, torch.float16, torch.bfloat16]
    restrict_contexts = [False, True, False]
    fasts = [False]*3

    for name, dtype, restrict_context_size, use_fast in zip(model_names, dtypes, restrict_contexts, fasts):
        try:
            main(name, dtype, restrict_context_size, use_fast)
        except RuntimeError as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                pass
            else:
                raise e
        
        gc.collect()
        torch.cuda.empty_cache()