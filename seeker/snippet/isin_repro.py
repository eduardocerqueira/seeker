#date: 2024-03-11T17:02:47Z
#url: https://api.github.com/gists/88c7660dab316c7a0702c87332eace62
#owner: https://api.github.com/users/Kh4L

import timeit
import numpy as np
import torch

def cantor_pairing_function(a: torch.tensor, b: torch.tensor) -> torch.tensor:
    return (a + b) * (a + b + 1) / 2 + a

def edge_isin_mem(combined_edge_index: torch.tensor, combined_memory: torch.tensor, use_isin: bool) -> torch.tensor:
    combined_memory, _ = torch.sort(combined_memory)
    if use_isin:
        edge_isin_mem_tensor = torch.isin(combined_edge_index, combined_memory)
    else:
        edge_isin_mem_tensor = torch.eq(combined_edge_index.unsqueeze(-1), combined_memory.view((1,1,-1))).any(-1)


def predict_link(query_src: np.ndarray,
                 query_dst: np.ndarray,
                 memory_tensor: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(query_src))
    idx = 0
    for src, dst in zip(query_src, query_dst):
        if (src, dst) in memory_tensor:
            pred[idx] = 1
        idx += 1
    return pred

def time_for_max(max_val):

    print(f"##################### max_val {max_val} #####################")

    setup_code = '''
from __main__ import edge_isin_mem
import torch
query_edge_indices = torch.tensor([[53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53],
    [512575,24797,460924,101881,76126,361755,160489,214872,110049,7832,111721,383104,297412,297991,567001,539476,417683,505760,421773,109465,240880]]).cuda()
memory_tensor = torch.randint(low=0, high={}, size=(2, 15966659)).cuda().float()
combined_edge_index = cantor_pairing_function(query_edge_indices[0, :],
                                              query_edge_indices[1, :])
combined_memory = cantor_pairing_function(memory_tensor[0, :],
                                          memory_tensor[1, :])
'''.format(max_val)

    x = True
    execution_times = timeit.repeat("edge_isin_mem(combined_edge_index, combined_memory, use_isin)", globals={**globals(), **locals()}, repeat=3, number=1000,setup=setup_code)
    average_time_isin = sum(execution_times) / len(execution_times)
    print(f"torch.isin avg time {max_val} : {average_time_isin} seconds")

    use_isin = False
    execution_times = timeit.repeat("edge_isin_mem(combined_edge_index, combined_memory, use_isin)", globals={**globals(), **locals()}, repeat=3, number=1000,setup=setup_code)
    average_time_torch = sum(execution_times) / len(execution_times)
    print(f"torch implem avg time {max_val} : {average_time_torch} seconds")

    setup_code = '''
from __main__ import edge_isin_mem
import torch
query_edge_indices = torch.tensor([[53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53],
    [512575,24797,460924,101881,76126,361755,160489,214872,110049,7832,111721,383104,297412,297991,567001,539476,417683,505760,421773,109465,240880]])
query_src = query_edge_indices[0].numpy()
query_dst = query_edge_indices[1].numpy()
memory_tensor = torch.randint(low=0, high={}, size=(2, 15966659)).numpy()
'''.format(max_val)


    execution_times = timeit.repeat("predict_link(query_src, query_dst, memory_tensor)", globals=globals(), repeat=3, number=1000, setup=setup_code)
    average_time_np = sum(execution_times) / len(execution_times)
    print(f"NP avg time : {average_time_np} seconds")
    print(f"torch.isin in {average_time_isin / average_time_np}x slower than NP")
    print(f"torch implem in {average_time_torch / average_time_np}x slower than NP")

values_list = [2**i for i in range(0, 24, 1)]
values_list = values_list + [574790]

for i in values_list:
    time_for_max(i)