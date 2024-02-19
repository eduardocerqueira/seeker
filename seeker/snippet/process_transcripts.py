#date: 2024-02-19T17:09:03Z
#url: https://api.github.com/gists/c19dbff531712ec1c5ee84e68d7785e4
#owner: https://api.github.com/users/bigsnarfdude

import sys
from llama_cpp import Llama
from prompt import PromptText
from util import read_file_to_string


input_file_path = sys.argv[1] if len(sys.argv) > 1 else sys.exit("No input file found")
output_file_path = sys.argv[2] if len(sys.argv) > 1 else sys.exit("No output file stated")

prompt = PromptText().summary_text
content = read_file_to_string(input_file_path)
model_path = "/Users/vincent/Downloads/mistral-7b-instruct-v0.2.Q4_0.gguf"
content_word_max = 20000
gpu_layers = 13
generated_response = 1000
summary_submission = prompt+content+'[/INST]'

llm = Llama(model_path=model_path, n_ctx=content_word_max, n_gpu_layers=gpu_layers)
output = "**********"=generated_response, stop=["</s>"])

with open(output_file_path, 'w') as output_file:
    output_file.write(f'\n\n\n<<<<<<<<<<<<<<<<<<<<<<<<   {output_file_path}   >>>>>>>>>>>>>>>>>>>>>>>\n')
    output_file.write(output['choices'][0]['text'])
choices'][0]['text'])
