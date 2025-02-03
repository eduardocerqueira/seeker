#date: 2025-02-03T16:33:46Z
#url: https://api.github.com/gists/a91e0eda855c5987806e54b2f385bd86
#owner: https://api.github.com/users/ashishsecdev

import requests
import json
import re

def query_ollama(prompt, model="deepseek-r1:1.5b"):
    url = "http://localhost:11434/api/generate"  # Ollama API endpoint
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload, stream=True)
        full_response = ""

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    part = data.get("response", "")
                    full_response += part

                    if data.get("done", False):
                        cleaned_response = full_response.replace("<think>", "").replace("</think>", "") # Remove annoying Think tag
                        print("\n===>", cleaned_response)
                        
                except json.JSONDecodeError:
                    print("Error: Invalid JSON in response line.")
                    print("Raw Line:", line.decode('utf-8'))
                    
    except requests.exceptions.RequestException as e:
        print("Error querying Ollama:", e)

if __name__ == "__main__":
    nichod_query = "Analyze the provided AWS CloudTrail logs and generate a comprehensive security summary, identifying key findings such as suspicious activities or potential vulnerabilities. Assess the severity based on its potential impact on the environment and infrastructure. Categorize each finding into one of the following priority levels: Critical, High, Medium, or Low, considering the exploitability, potential risk, and overall threat severity..\n"
    user_input = ""
    print("Enter AWS Cloudtrail JSON:")

    while True:
        line = input()
        if line == "":
            break
        user_input += line + "\n"

    cleaned_user_input = re.sub(r'[\s\n\t]+', '', user_input) #Removes newlines and tabs from user input as JSON breaks a times
    final_input = nichod_query + cleaned_user_input
    print("Final input:", repr(final_input)) #Prints the final userinput + predefined prompts

    query_ollama(final_input)
