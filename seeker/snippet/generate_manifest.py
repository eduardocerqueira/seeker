#date: 2025-08-25T17:08:20Z
#url: https://api.github.com/gists/1010989d4e976fe26d1a5d4b2c533cff
#owner: https://api.github.com/users/ayhanasghari

import os

def extract_description(file_path):
    """خواندن توضیح از داک‌استرینگ یا کامنت‌های ابتدایی فایل"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[:10]:  # فقط خطوط ابتدایی
                line = line.strip()
                if line.startswith('"""') or line.startswith("'''"):
                    return line.strip('"""').strip("'''").strip()
                elif line.startswith('#'):
                    return line.lstrip('#').strip()
        return "بدون توضیح"
    except Exception as e:
        return f"خطا در خواندن: {e}"

def scan_project(root_dir="."):
    manifest = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py") and filename != "generate_manifest.py":
                rel_path = os.path.relpath(os.path.join(dirpath, filename), root_dir)
                description = extract_description(os.path.join(dirpath, filename))
                manifest[rel_path.replace("\\", "/")] = description
    return manifest

def save_manifest(manifest, output_file="project_manifest.py"):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# فایل مانیفست خودکار\n\n")
        f.write("PROJECT_FILES = {\n")
        for path, desc in manifest.items():
            f.write(f'    "{path}": "{desc}",\n')
        f.write("}\n")

if __name__ == "__main__":
    manifest = scan_project()
    save_manifest(manifest)
    print("✅ مانیفست با موفقیت ساخته شد: project_manifest.py")
