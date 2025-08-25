#date: 2025-08-25T17:08:20Z
#url: https://api.github.com/gists/1010989d4e976fe26d1a5d4b2c533cff
#owner: https://api.github.com/users/ayhanasghari

import os

def scan_python_files(root_dir="E:/AI_arshad/Term4/AI_02/tabriz_rl_sim/"):
    manifest = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, root_dir)
                manifest.append(rel_path)
    return manifest

if __name__ == "__main__":
    print("📦 Python Files in Project Directory:\n")
    files = scan_python_files()
    for i, f in enumerate(sorted(files), 1):
        print(f"{i:02d}. {f}")
