#date: 2023-06-05T17:09:55Z
#url: https://api.github.com/gists/9b12fcb96834435aaa66d1affd73c0cc
#owner: https://api.github.com/users/joshfinley

import sys
import base64

def is_base64(s):
    try:
        # Attempt to decode the string as base64
        base64.b64decode(s)
        return True
    except:
        return False

def base64_encode_file(file_path):
    try:
        with open(file_path, 'r') as file:
            contents = file.read()
            if not is_base64(contents):
                # If contents are plain text, encode them
                encoded = base64.b64encode(contents.encode()).decode()
                with open(file_path, 'w') as encoded_file:
                    encoded_file.write(encoded)
                    print(f"File {file_path} base64 encoded successfully.")
            else:
                print(f"File {file_path} is already base64 encoded.")
    except FileNotFoundError:
        print(f"File {file_path} not found.")

def base64_decode_file(file_path):
    try:
        with open(file_path, 'r') as file:
            contents = file.read()
            if is_base64(contents):
                # If contents are base64 encoded, decode them
                decoded = base64.b64decode(contents).decode()
                with open(file_path, 'w') as decoded_file:
                    decoded_file.write(decoded)
                    print(f"File {file_path} base64 decoded successfully.")
            else:
                print(f"File {file_path} is not base64 encoded.")
    except FileNotFoundError:
        print(f"File {file_path} not found.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python base64_converter.py <encode|decode> <file_path>")
        sys.exit(1)

    action = sys.argv[1]
    file_path = sys.argv[2]

    if action == "encode":
        base64_encode_file(file_path)
    elif action == "decode":
        base64_decode_file(file_path)
    else:
        print("Invalid action. Use 'encode' or 'decode'.")
