#date: 2024-05-08T16:49:58Z
#url: https://api.github.com/gists/ce1adfad1d561c1e8dc92666ab5a9e8c
#owner: https://api.github.com/users/Cyrilvallez

# TO CREATE FIGURES OF THE BENCHMARK 1

import json
import matplotlib.pyplot as plt
import numpy as np

model_names = ['Mistral-7B-v0.1', 'Llama-2-7b-hf', 'Meta-Llama-3-8B']
fix_batch = '_batch_size_1_input_size_300'
fix_length = "**********"
folder = 'benchmark/'

fig_folder = 'results/'


def load_json(filename: str) -> dict:
    with open(filename, 'r') as fp:
        data = json.load(fp)
    data = {int(k): v for k,v in data.items()}
    a, b = np.array(list(data.keys())), np.array(list(data.values()))
    sorting = np.argsort(a)
    return a[sorting], b[sorting]
    
    
    
# Fix batch figures

for model in model_names:
    # memory figure
    plt.figure()
    plt.plot(*load_json(folder + 'legacy_' + model + fix_batch + '_memory.json'), 'b-', label='Before')
    plt.plot(*load_json(folder + 'patched_' + model + fix_batch + '_memory.json'), 'r-', label='After')
    plt.xlabel('New tokens generated')
    plt.ylabel('Peak memory usage [GiB]')
    plt.grid()
    plt.legend()
    plt.title(model + '\nInput size 300, Batch size 1')
    plt.savefig(fig_folder + model + '_memory_fix_batch.pdf', bbox_inches='tight')
    plt.show()


    # time figure
    plt.figure()
    plt.plot(*load_json(folder + 'legacy_' + model + fix_batch + '_time.json'), 'b-', label='Before')
    plt.plot(*load_json(folder + 'patched_' + model + fix_batch + '_time.json'), 'r-', label='After')
    plt.xlabel('New tokens generated')
    plt.ylabel('Generation time [s]')
    plt.grid()
    plt.legend()
    plt.title(model + '\nInput size 300, Batch size 1')
    plt.savefig(fig_folder + model + '_time_fix_batch.pdf', bbox_inches='tight')
    plt.show()
    
    
    
# Fix new tokens figures

for model in model_names:

    # memory figure
    plt.figure()
    plt.plot(*load_json(folder + 'legacy_' + model + fix_length + '_memory.json'), 'b-', label='Before')
    plt.plot(*load_json(folder + 'patched_' + model + fix_length + '_memory.json'), 'r-', label='After')
    plt.xlabel('Batch size')
    plt.ylabel('Peak memory usage [GiB]')
    plt.grid()
    plt.legend()
    plt.title(model + '\nInput size 300, New tokens 2000')
    if model == model_names[1]:
        plt.text(4.1, 9, 'OOM after this point', color='b')
    plt.savefig(fig_folder + model + '_memory_fix_length.pdf', bbox_inches='tight')
    plt.show()


    # time figure
    plt.figure()
    plt.plot(*load_json(folder + 'legacy_' + model + fix_length + '_time.json'), 'b-', label='Before')
    plt.plot(*load_json(folder + 'patched_' + model + fix_length + '_time.json'), 'r-', label='After')
    plt.xlabel('Batch size')
    plt.ylabel('Generation time [s]')
    plt.grid()
    plt.legend()
    plt.title(model + '\nInput size 300, New tokens 2000')
    if model == model_names[1]:
        plt.text(4.1, 57, 'OOM after this point', color='b')
    plt.savefig(fig_folder + model + '_time_fix_length.pdf', bbox_inches='tight')
    plt.show()ight')
    plt.show()