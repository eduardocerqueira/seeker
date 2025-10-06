#date: 2025-10-06T16:58:25Z
#url: https://api.github.com/gists/188ad5a3a7c77ab880adcd07695f3d69
#owner: https://api.github.com/users/maksimkunaev

import requests
import os
import tempfile

# Указываем директорию для временных буферов на workspace
temp_dir = "/workspace/tmp"
os.makedirs(temp_dir, exist_ok=True)
tempfile.tempdir = temp_dir

files = [
    ("https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF/resolve/main/Q8_0/Qwen3-235B-A22B-Q8_0-00005-of-00006.gguf", 
     "/workspace/Qwen3-235B-A22B-Q8/Qwen3-235B-A22B-Q8_0-00005-of-00006.gguf"),
    ("https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF/resolve/main/Q8_0/Qwen3-235B-A22B-Q8_0-00006-of-00006.gguf",
     "/workspace/Qwen3-235B-A22B-Q8/Qwen3-235B-A22B-Q8_0-00006-of-00006.gguf")
]

for url, output_path in files:
    print(f"Downloading {os.path.basename(output_path)}...")

    # Если файл уже есть, продолжаем с места остановки
    headers = {}
    if os.path.exists(output_path):
        existing_size = os.path.getsize(output_path)
        headers['Range'] = f"bytes={existing_size}-"
        mode = 'ab'
        print(f"  Resuming from {existing_size / (1024**3):.2f} GB")
    else:
        mode = 'wb'

    # Используем stream=True и записываем напрямую в файл на workspace
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        if response.status_code not in [200, 206]:
            print(f"  ✗ Error {response.status_code}")
            continue

        with open(output_path, mode) as f:
            for chunk in response.iter_content(chunk_size=100*1024*1024):  # 100 MB
                if chunk:
                    f.write(chunk)

    print(f"  ✓ Done: {output_path}")
