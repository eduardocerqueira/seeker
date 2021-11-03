#date: 2021-11-03T17:01:30Z
#url: https://api.github.com/gists/f9a0b923439395440407e9f056b65043
#owner: https://api.github.com/users/geryxyz

import argparse
import sys


def main():
    """
    python wtf-abbr.py --abbr WTF --long "would you like to find the perfect abbreviation"
    WTF abbreviation found!
    Would you like To Find the perfect abbreviation
    ^              ^  ^
    """

    parser = argparse.ArgumentParser(prog='wtf-abbr', description='Would you like To Find the perfect abbreviation?')
    parser.add_argument('--abbr', type=str,
                        help='the abbreviation you are looking for')
    parser.add_argument('--long', type=str,
                        help='the long form of the name')

    args = parser.parse_args()
    long_name: str = args.long.lower()
    abbreviation: str = args.abbr.lower()

    selected_indexes = []
    for index, letter in enumerate(abbreviation):
        selected_indexes.append(long_name.find(letter))

    result = ''
    marks = ''
    for index, letter in enumerate(long_name):
        if index in selected_indexes:
            result += letter.upper()
            marks += '^'
        else:
            result += letter
            marks += ' '

    all_letter_present = -1 not in selected_indexes
    is_in_correct_order = sorted(selected_indexes) == selected_indexes
    if all_letter_present and is_in_correct_order:
        print(f"WTF abbreviation found!\n{result}\n{marks}")
    else:
        print(f"No suitable selection.")
        if not all_letter_present:
            print("Some letters are missing.")
        if not is_in_correct_order:
            print("The letters are not in the correct order.")
        print(f"The best we could do is the following.\n{result}\n{marks}")
        sys.exit(1)


if __name__ == '__main__':
    main()
