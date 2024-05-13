#date: 2024-05-13T16:47:40Z
#url: https://api.github.com/gists/01bcf2d136b62a0f01a70343c76b14af
#owner: https://api.github.com/users/akoksal

from vllm import LLM, SamplingParams
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="Model name")
parser.add_argument("--gpu_count", type=int, help="GPU count")

args = parser.parse_args()
model_name = args.model_name
gpu_count = args.gpu_count

CACHE_DIR = "hidden"

## Load the model
llm = LLM(model=model_name,
          download_dir=CACHE_DIR,
          tensor_parallel_size=gpu_count, 
          gpu_memory_utilization=0.85,
          trust_remote_code=True)

## Load the data
with open("gsm8k.jsonl", "r") as f:
    examples = [json.loads(x) for x in f]

questions = [ex["question"] for ex in examples][1:101]

shot = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72


Question: {question}
Answer:"""

prompts = []
for question in questions:
    prompts.append(shot.format(question=question))


## Generation
sampling_params = SamplingParams(temperature=0.7, max_tokens=256, stop=["\nQuestion: "**********"
llm_output = llm.generate(prompts, sampling_params)
outputs = [o.outputs[0].text for o in llm_output]

## Save the outputs
mn_formatted = model_name.replace('/', '_')
with open(f"GSM8K/{mn_formatted}.json", "w") as f:
    json.dump(outputs, f, indent=2) json.dump(outputs, f, indent=2)