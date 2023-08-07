#date: 2023-08-07T16:52:48Z
#url: https://api.github.com/gists/30c64ff7c6c84e5a72ae965c027182f2
#owner: https://api.github.com/users/mxyng

import sys
import json
import argparse
import requests


def generate(name, prompt, context=[], host='127.0.0.1', port=11434, model='llama2'):
    r = requests.post(f'http://{host}:{port}/api/generate', json={'model': model, 'prompt': prompt, 'context': context}, stream=True)
    r.raise_for_status()

    lines = []

    print(f'>>> {name}: ', end='', file=sys.stderr, flush=True)
    for line in r.iter_lines():
        body = json.loads(line)
        text = body.get('response', '')
        print(text, end='', file=sys.stderr, flush=True)
        lines.append(text)

        if body.get('done', False):
            body['response'] = ''.join(lines)
            print(file=sys.stderr, flush=True)
            return body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama2')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=11434)
    parser.add_argument('prompt', help='initial prompt')
    args = parser.parse_args()

    prompt = args.prompt
    print(f'>>> prompt: {prompt}', file=sys.stderr)

    args = vars(args)
    del(args['prompt'])

    alice_context = []
    bob_context = []
    while True:
        alice_response = generate('alice', prompt, context=alice_context, **args)
        alice_context = alice_response.get('context', [])
        prompt = alice_response.get('response', '')

        bob_response = generate('bob', prompt, context=bob_context, **args)
        bob_context = bob_response.get('context', [])
        prompt = bob_response.get('response', '')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
