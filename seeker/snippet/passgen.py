#date: 2025-09-02T16:55:35Z
#url: https://api.github.com/gists/4442ac5aef03ff00450b50737eb5ca56
#owner: https://api.github.com/users/tsoushi

import argparse
import random

alpha = 'abcdefghijklmnopqrstuvwxyz'
number = '0123456789'
special = '!@#$%^&*()-+'


parser = argparse.ArgumentParser()

parser.add_argument('--length', '-l', type=int, default=32, help='パスワードの長さ')
parser.add_argument('--only-upper', action='store_true', help='大文字アルファベットのみを使用する')
parser.add_argument('--no-alpha', action='store_true', help='英字を使用しない')
parser.add_argument('--no-upper', action='store_true', help='大文字アルファベットを使用しない')
parser.add_argument('--no-number', action='store_true', help='数字を使用しない')
parser.add_argument('--no-special', action='store_true', help='特殊文字を使用しない')

args = parser.parse_args()


candidate = ''
if not args.no_alpha:
    if args.only_upper:
        candidate += alpha.upper()
    elif args.no_upper:
        candidate += alpha
    else:
        candidate += alpha + alpha.upper()
if not args.no_number:
    candidate += number
if not args.no_special:
    candidate += special

if not candidate:
    parser.error('少なくとも1つの文字セットを選択してください。')
    exit(1)

password = "**********"
print(password)ate) for _ in range(args.length))
print(password)