#date: 2025-09-04T16:54:04Z
#url: https://api.github.com/gists/e7120d891daa7971355d5f378ec4c74d
#owner: https://api.github.com/users/zeozeozeo

# Run this script inside your Discord data package:
# https://support.discord.com/hc/en-us/articles/360004027692-Requesting-a-Copy-of-your-Data

import json
import os
import re
import random
import argparse
from collections import defaultdict, Counter

class MarkovChain:
    def __init__(self, state_size=2):
        self.state_size = state_size
        self.chain = defaultdict(Counter)
        self.start_words = []
        
    def build_chain(self, normalized_messages):
        for message in normalized_messages:
            words = message.split()
            if len(words) <= self.state_size:
                continue

            self.start_words.append(tuple(words[:self.state_size]))

            for i in range(len(words) - self.state_size):
                current_state = tuple(words[i:i+self.state_size])
                next_word = words[i+self.state_size]
                self.chain[current_state][next_word] += 1
    
    def generate_sentence(self, max_length=25, start_with=None):
        if not self.chain or not self.start_words:
            return 'error: no messages'
        
        if start_with:
            start_words = normalize_text(start_with).split()
            if len(start_words) >= self.state_size:
                current_state = tuple(start_words[:self.state_size])
                if current_state not in self.chain:
                    current_state = random.choice(self.start_words)
            else:
                possible_starts = [state for state in self.start_words if state[0] == start_words[0]]
                if possible_starts:
                    current_state = random.choice(possible_starts)
                else:
                    print("WARN: couldn't find a starting sequence, generating garbage")
                    current_state = random.choice(self.start_words)
        else:
            current_state = random.choice(self.start_words)
        
        sentence = list(current_state)
        
        for _ in range(max_length - self.state_size):
            if current_state not in self.chain:
                break

            next_words_counter = self.chain[current_state]
            next_word = random.choices(
                list(next_words_counter.keys()),
                weights=list(next_words_counter.values()),
                k=1
            )[0]
            
            sentence.append(next_word)
            current_state = tuple(sentence[-self.state_size:])

            if next_word.endswith(('.', '!', '?')) and random.random() < 0.3:
                break

        return ' '.join(sentence).capitalize()

def parse_discord_data(path):
    messages = []
    messages_dir = os.path.join(path, 'Messages')
    
    if not os.path.exists(messages_dir):
        raise FileNotFoundError(f'Messages directory not found at {messages_dir}')

    for channel_dir in os.listdir(messages_dir):
        channel_path = os.path.join(messages_dir, channel_dir)
        if os.path.isdir(channel_path):
            messages_file = os.path.join(channel_path, 'messages.json')
            if os.path.exists(messages_file):
                try:
                    with open(messages_file, 'r', encoding='utf-8') as f:
                        channel_data = json.load(f)
                    for message in channel_data:
                        if message.get('Contents') and message['Contents'].strip():
                            messages.append(message['Contents'])
                except Exception as e:
                    print(f"couldn't parse {messages_file}: {e}")
    return messages

def normalize_text(text):
    if not text:
        return ''
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s.\?!,]', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text

def normalize_all(messages):
    return [normalized for msg in messages if (normalized := normalize_text(msg))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path to Discord data package', default='.', nargs='?')
    parser.add_argument('--state-size', type=int, default=2)
    
    args = parser.parse_args()

    try:
        print('loading discord data package...')
        raw_messages = parse_discord_data(args.path)
        print(f'parsed {len(raw_messages)} messages')
        
        print('normalizing text...')
        normalized_messages = normalize_all(raw_messages)
        print(f'filtered {len(normalized_messages)} messages')
        
        if not normalized_messages:
            print('no messages!')
            return

        print('building markov chain...')
        chain = MarkovChain(state_size=args.state_size)
        chain.build_chain(normalized_messages)

        print('\n'+'='*50)

        while True:
            inp = input('\nyou: ').strip()
            if inp.lower() in ['quit', 'exit', 'q']:
                break
            elif inp.lower() in ['generate', 'g']:
                response = chain.generate_sentence()
                print(f'also you: {response}')
            elif inp:
                response = chain.generate_sentence(start_with=inp)
                print(f'also you: {response}')
    except FileNotFoundError as e:
        print(f'error: {e}')
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
