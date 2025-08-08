#date: 2025-08-08T16:51:44Z
#url: https://api.github.com/gists/269ec86255cb8c6434929630db83c115
#owner: https://api.github.com/users/datavudeja

import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        return result

file_path = 'data/sample.csv'
encoding_info = detect_encoding(file_path)
print(f"Detected encoding: {encoding_info['encoding']}")