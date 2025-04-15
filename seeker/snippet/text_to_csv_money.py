#date: 2025-04-15T17:04:53Z
#url: https://api.github.com/gists/ea5c96c26c6a982cd25ed47599700320
#owner: https://api.github.com/users/TylerCode

def process_bonk_file(input_file="bonk.txt", output_file="bonk_fixed.csv"):
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    
    processed_lines = []
    for line in lines:
        line = line.rstrip()
        if line.startswith('$'):
            # Add comma to CSVify it and toss the $ otherwise excel will be a baby
            processed_lines.append(line.replace('$','') + ',\n')
    
    try:
        with open(output_file, 'w') as f:
            f.writelines(processed_lines)
        print(f"Successfully processed {len(processed_lines)} lines and saved to {output_file}")
    except Exception as e:
        print(f"Error writing to output file: {e}")
        return

if __name__ == "__main__":
    process_bonk_file()
