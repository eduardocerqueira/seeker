#date: 2024-03-15T17:02:03Z
#url: https://api.github.com/gists/6507b6ec8d00047e586ae867ed62a9f7
#owner: https://api.github.com/users/watanany

from toolz import get_in

def get_role(v): 
    return get_in(['message', 'author', 'role'], v, '')

def get_message(v):
    return get_in(['message', 'content', 'parts', 0], v, '')

def main():
    with open('data/conversations.json') as r:
        conversations = json.load(r)
    
    for conversation in conversations:
        print(f'# {conversation["title"]}\n\n\n\n', end='')
        for role, msg in [(get_role(v), m) for v in conversation['mapping'].values() if (m := get_message(v)) != '']:
            print(f'## {role}')
            print(msg)
            print('\n\n\n', end='')
            
if __name__ == '__main__':
    main()
    