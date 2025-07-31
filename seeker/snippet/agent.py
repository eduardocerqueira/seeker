#date: 2025-07-31T16:54:40Z
#url: https://api.github.com/gists/3f132453e611d0d78ac80f790730ecc9
#owner: https://api.github.com/users/joaomelosb

import json
from openai import OpenAI
from random import choice

def get_color(exclude):
    colors = [
        'red',
        'green',
        'blue',
        'yellow'
    ]
    
    print(f'log > model called tool with "{exclude}" as input')
    
    exclude = [color.strip().lower() for color in exclude.split(',')]
    
    return choice([color for color in colors if color not in exclude])

client = OpenAI()
MODEL = 'gpt-4.1-nano'

messages = [
    {'role': 'system', 'content': "You're a helpful assistant"},
]
tools = [
    {
        'type': 'function',
        'name': 'get_color',
        'description': 'Get a random color, excluding colors included in input',
        'parameters': {
            'type': 'object',
            'properties': {
                'exclude': {
                    'type': 'string',
                    'description': 'A comma separated list of colors to exclude'
                }
            },
            'required': ['exclude'],
            'additionalProperties': False
        },
        'strict': True
    }
]
skip_input = False

while True:
    if not skip_input:
        try:
            text = input('user > ')
            
            if text in ('quit', 'exit'):
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            break
        
        messages.append({'role': 'user', 'content': text})
    
    output = client.responses.create(
        model = MODEL,
        input = messages,
        tools = tools
    ).output[0]
    
    if output.type == 'function_call':
        messages.append(output)
        messages.append({
            'type': 'function_call_output',
            'call_id': output.call_id,
            'output': get_color(json.loads(output.arguments)['exclude'])
        })
        skip_input = True
    else:
        print(f'assistant > {output.content[0].text}')
        skip_input = False