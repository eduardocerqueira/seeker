#date: 2024-05-08T17:05:07Z
#url: https://api.github.com/gists/8344f35ace2b6bff0c83a8dcaf7dd167
#owner: https://api.github.com/users/MicBosi

# How to run:
# python3 ollama_bench.py
# python3 ollama_bench.py --runs 5 --model llama3:8b --prompt "Why is the sky blue? Use exactly 200 words." --print_llm_response

import http.client
import json
import time
import platform

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, help="Number of runs to average over", default=5)
parser.add_argument("--model", type=str, help="Model name", default="llama3:8b")
parser.add_argument("--prompt", type=str, help="Prompt", default="Why is the sky blue? Use exactly 200 words.")
parser.add_argument("--print_llm_response", action="store_true", help="Print LLM response")
args = parser.parse_args()

runs = args.runs
model = args.model
prompt = args.prompt
print_llm_response = args.print_llm_response

def make_request():
    # Connect to the server
    conn = http.client.HTTPConnection('127.0.0.1', 11434)
    # Prepare the data to be sent in the POST request
    data = {
        "stream": False,
        "model": model,
        "prompt": prompt,
    }
    json_data = json.dumps(data)

    # Start timing
    start_time = time.time()

    # Send the POST request
    conn.request("POST", "/api/generate", body=json_data, headers={"Content-Type": "application/json"})

    # Get the response from the server
    response = conn.getresponse()
    response_data = response.read().decode()

    # End timing
    end_time = time.time()

    # Close the connection
    conn.close()

    # Calculate duration
    duration = end_time - start_time

    # Parse response JSON
    response_json = json.loads(response_data)

    # Print full json response
    # print(response_json)

    # Check if 'error' is set in the response then print the error and exit
    if 'error' in response_json:
        print(f"Error: {response_json['error']}")
        exit(1)

    # Count words in the response
    words = len(response_json['response'].split())

    # Calculate words per second
    words_per_second = words / duration

    return words, duration, words_per_second, response_json['response']

# Print model and prompt used
print(f"Model: {model}")
print(f"Prompt: {prompt}")
# Print system information like cpu, gpu and memory using the platform module
print(f"System: {platform.platform()}\n")

# List to store stats for each run
results = []

# Warm-up run
result = make_request()
print(f"Words: {result[0]}, Time: {result[1]:.2f}s, Words/Sec: {result[2]:.2f} (warm-up)\n")

# Call the function 5 times and gather stats
for _ in range(runs):
    result = make_request()
    results.append(result)
    print(f"Words: {result[0]}, Time: {result[1]:.2f}s, Words/Sec: {result[2]:.2f}")
    if print_llm_response:
        print(f"> {result[3]}\n")

# Calculate averages
average_words = sum(result[0] for result in results) / 5
average_time = sum(result[1] for result in results) / 5
average_wps = sum(result[2] for result in results) / 5

print(f"\nAverage Words: {average_words}, Average Time: {average_time:.2f}s, Average Words/Sec: {average_wps:.2f}")
