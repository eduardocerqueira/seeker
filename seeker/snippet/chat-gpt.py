#date: 2023-06-06T17:01:18Z
#url: https://api.github.com/gists/afe46feaedd512676fb5bb22ab33cf49
#owner: https://api.github.com/users/keyehzy

import os
import requests
import json
import tiktoken
import time

# curl https://api.openai.com/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -d '{
#     "model": "gpt-3.5-turbo",
#     "messages": [{"role": "user", "content": "Hello!"}]
#   }'

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
LOGGING = True

model = "gpt-3.5-turbo"
system_message = {"role": "system", "content": "You are a helpful AGI."}
max_response_tokens = "**********"
token_limit = "**********"
conversation=[]
conversation.append(system_message)

 "**********"d "**********"e "**********"f "**********"  "**********"n "**********"u "**********"m "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"_ "**********"f "**********"r "**********"o "**********"m "**********"_ "**********"m "**********"e "**********"s "**********"s "**********"a "**********"g "**********"e "**********"s "**********"( "**********"m "**********"e "**********"s "**********"s "**********"a "**********"g "**********"e "**********"s "**********", "**********"  "**********"m "**********"o "**********"d "**********"e "**********"l "**********"= "**********"" "**********"g "**********"p "**********"t "**********"- "**********"3 "**********". "**********"5 "**********"- "**********"t "**********"u "**********"r "**********"b "**********"o "**********"" "**********") "**********": "**********"
    encoding = "**********"
    num_tokens = "**********"
    for message in messages:
        num_tokens += "**********"
        for key, value in message.items():
            num_tokens += "**********"
            if key == "name":
                num_tokens += "**********"
    num_tokens += "**********"
    return num_tokens

def get_completion(history, model = 'gpt-3.5-turbo'):
    response = requests.post(
        url = 'https://api.openai.com/v1/chat/completions',
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {OPENAI_API_KEY}',
        },
        data = json.dumps({
            'model': model,
            'messages': conversation,
            'temperature': 0.7,
            'max_tokens': "**********"
        }),
    )
    payload = json.loads(response.text)
    return payload['choices'][0]['message']['content']

while(True):
    user_input = input('User:')
    conversation.append({"role": "user", "content": user_input})
    conv_history_tokens = "**********"

 "**********"  "**********"  "**********"  "**********"  "**********"w "**********"h "**********"i "**********"l "**********"e "**********"  "**********"( "**********"c "**********"o "**********"n "**********"v "**********"_ "**********"h "**********"i "**********"s "**********"t "**********"o "**********"r "**********"y "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"+ "**********"m "**********"a "**********"x "**********"_ "**********"r "**********"e "**********"s "**********"p "**********"o "**********"n "**********"s "**********"e "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"  "**********"> "**********"= "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"l "**********"i "**********"m "**********"i "**********"t "**********") "**********": "**********"
        del conversation[1] 
        conv_history_tokens = "**********"

    completion = get_completion(conversation, model)
    conversation.append({"role": "assistant", "content": completion})
    print(f'\nAGI: {completion}')

    if LOGGING:
        with open('conversation.txt', 'w') as f:
            json.dump(conversation, f)
