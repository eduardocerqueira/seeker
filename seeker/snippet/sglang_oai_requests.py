#date: 2025-05-09T17:06:44Z
#url: https://api.github.com/gists/0b8d12b0115bc8e50ee5f400d3910d01
#owner: https://api.github.com/users/Lyken17

import pandas as pd
import time
from sglang import gen, system
import os, sys
import asyncio
import openai
from tqdm import tqdm
import json
# from sglang.utils import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process
port = 30000
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

# Read the parquet file
# fpath = f'train.parquet'
fpath = f"simple_r1_train.parquet"
prefix = fpath.split('.')[0]

# Initialize the model client
model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
suffix = model.split('/')[-1]
# python -m sglang.launch_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --trust-remote-code --dp-size 2 --tp-size 4

print(f"Waiting for server to start on port {port}")
wait_for_server(f"http://localhost:{port}")
print(f"Server started on http://localhost:{port}")
# Check if the client is healthy and alive
try:
    health_check = client.models.list()
    print("\nClient Health Check:")
    print("-------------------")
    print(f"Connection to port {port} successful")
    print(f"Available models: {[model.id for model in health_check.data]}")
    print("Client is healthy and ready to process requests")
except Exception as e:
    print("\nClient Health Check Failed:")
    print("-------------------------")
    print(f"Error connecting to server at port {port}: {str(e)}")
    print("Please ensure the server is running and accessible")
    # sys.exit(0)
    
    server_process, port = launch_server_cmd(
        f"python -m sglang.launch_server --model {model} --trust-remote-code --dp-size 2 --tp-size 4"
    )
    wait_for_server(f"http://localhost:{port}")
    client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")
    print(f"Server started on http://localhost:{port}")


df = pd.read_parquet(fpath)
if os.path.exists(f'{prefix}_with_{suffix}-generated.parquet'):
    print("Loading existing generated responses")
    df = pd.read_parquet(f'{prefix}_with_{suffix}-generated.parquet')

 # Add the generated response to the dataframe in a new column
if f'{suffix}-generated' not in df.columns:
    df[f'{suffix}-generated'] = None
if f'{suffix}-logprobs' not in df.columns:
    df[f'{suffix}-logprobs'] = None


progress = tqdm(total=len(df), desc="Data Generation progress")  # Initialize progress bar

def generate_response(index, question):
    print(f"[filled {model}] Question {index} / {len(df)}: ")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": question},
        ],
        temperature=0.6,
        max_tokens= "**********"
        # n=8,
        logprobs=True,
    )
    message_content = [choice.message.content for choice in response.choices]
    log_probs = json.dumps([[logprobs.logprob for logprobs in choice.logprobs.content] for choice in response.choices])
    
    df.at[index, f'{suffix}-generated'] = message_content
    df.at[index, f'{suffix}-logprobs'] = log_probs
    
    # await asyncio.sleep(10)
    print("=" * 50)
    print(f"Question {index} / {len(df)}: {question}")
    print(f"Response: done")
    # if index % 5 == 0 or index == len(df) - 1:
    #     # pandas is thread safe
    #     df.to_parquet('train_with_qwen32b-r1-generated.parquet')
    return index, question,message_content

import concurrent.futures

def save_df(df, output_path):
    output_path_temp = output_path + '.temp'
    df.to_parquet(output_path_temp)
    os.rename(output_path_temp, output_path)

def main():
    print("\nProcessing question with the model:")
    print("----------------------------------")
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        tasks = []
        for index, row in df.iterrows():
            # Get the question text and clean it
            if row[f'{suffix}-generated'] is not None:
                print(f"[skip] Question {index} / {len(df)}: already generated")
                progress.update(1)
                continue
            question = row['question'].strip()
            # print(f"[feed to Queue] Question {index} / {len(df)}: ")
            tasks.append(executor.submit(generate_response, index, question))

        for future in concurrent.futures.as_completed(tasks):
            index, question, message_content = future.result()
            print(f"Question {index} / {len(df)}: {question}")
            print(f"Response: {message_content}")
            if index % 5 == 0 or index == len(df) - 1:
                # pandas is thread safe
                # df.to_parquet(f'{prefix}_with_{suffix}-generated.parquet')
                save_df(df, f'{prefix}_with_{suffix}-generated.parquet')
            progress.update(1)
        
        # df.to_parquet(f'{prefix}_with_{suffix}-generated.parquet')
        save_df(df, f'{prefix}_with_{suffix}-generated.parquet')
if __name__ == "__main__":
    main()
