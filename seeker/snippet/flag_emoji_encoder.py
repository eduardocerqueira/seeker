#date: 2025-07-18T16:58:33Z
#url: https://api.github.com/gists/f1bb40f097c4262ba5c9c9f57c53424c
#owner: https://api.github.com/users/jyapayne

import argparse
import sys

def text_to_flag(text: str) -> str:
    """
    Convert input text to flag emojis by mapping A–Z/a–z to regional indicator
    symbols. Non-alphabetic characters are left unchanged.
    """
    result = []
    for char in text:
        if char.isalpha():
            # Regional indicator symbols start at U+1F1E6 ("A")
            base = ord(char.upper()) - ord('A')
            result.append(chr(0x1F1E6 + base))
        else:
            result.append(char)
    return ''.join(result)


def main():
    parser = argparse.ArgumentParser(description='Convert text to flag emojis')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--text', type=str, help='Text to convert directly')
    group.add_argument('-f', '--file', type=str, help='File to read text from')
    
    args = parser.parse_args()
    
    if args.text:
        text_input = args.text
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text_input = f.read().strip()
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file '{args.file}': {e}", file=sys.stderr)
            sys.exit(1)
    
    print(text_to_flag(text_input))


if __name__ == "__main__":
    main()

# Usage examples:
# python emoji.py -t "Hello World"
# python emoji.py --text "Hello World"
# python emoji.py -f input.txt
# python emoji.py --file input.txt
# python emoji.py -h  # Show help