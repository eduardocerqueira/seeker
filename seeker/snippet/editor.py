#date: 2024-05-02T16:59:31Z
#url: https://api.github.com/gists/4be889c448a26b7c4df669af4d496496
#owner: https://api.github.com/users/tyschacht

import subprocess
import os
import random
import time

def edit(contents: str):
    """
    Opens TextEdit on macOS and waits until it is closed to proceed.
    """
    # Get the current working directory
    current_dir = os.getcwd()
    # Generate a random number to include in the filename
    random_number = random.randint(1000, 9999)
    temp_file_path = os.path.join(current_dir, f'tempfile_{random_number}.json')

    # Create and close the temporary file explicitly
    with open(temp_file_path, 'w+') as tmp:
        tmp.write(contents)
        tmp.flush()

    # Change the file permissions to make it readable and writable by everyone
    os.chmod(temp_file_path, 0o666)

    # Introduce a delay
    time.sleep(1)  # Wait for 1 second before opening the file in Editor

    # Open the default text editor and wait for it to close
    editor_process = subprocess.Popen(['open', '-W', '-n', '-a', 'TextEdit', temp_file_path])

    # Wait for the TextEdit process to close
    editor_process.wait()

    # Read the modified content from the file
    with open(temp_file_path, 'r') as file:
        modified_content = file.read()

    # Clean up by removing the temporary file
    os.remove(temp_file_path)

    return modified_content

# Example usage:
if __name__ == "__main__":
    sample_contents = "How are you doing this. Tell me more about it:"
    modified_config = edit(sample_contents)
    print(modified_config)
