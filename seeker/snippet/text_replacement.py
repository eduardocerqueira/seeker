#date: 2021-11-09T17:03:33Z
#url: https://api.github.com/gists/dec4c11244b5f3bbe8f4e40cd7bdd1bb
#owner: https://api.github.com/users/PC-CNT

"""text_replacement.py"""
import sys
import os


try:
    input_txt = sys.argv[1]
    search_word = sys.argv[2]
    replace_word = sys.argv[3]

    text_dir = os.path.dirname(input_txt)
    text_name = os.path.splitext(os.path.basename(input_txt))[0]
    replace_text_path = os.path.join(text_dir, text_name + "_replace.txt")
    # print(replace_text_path)

    with open(input_txt, 'r') as f:
        source_text = f.read()
        # print(source_text.replace(search_word, replace_word))
    with open(replace_text_path, 'w') as f:
        f.write(source_text.replace(search_word, replace_word))
except IndexError as e:
    print("Error! : " + str(e))
    print("Usage: text_replacement.py <input_file> <search_word> <replace_word>")
    sys.exit(1)