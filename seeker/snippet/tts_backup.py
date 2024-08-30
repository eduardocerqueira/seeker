#date: 2024-08-30T16:44:48Z
#url: https://api.github.com/gists/fabf3408c00b3d37464248378f5decbf
#owner: https://api.github.com/users/DerKleineLi

import concurrent.futures
import json
import mimetypes
import re
import shutil
import sys
from pathlib import Path

import magic
import requests

URL_FOLDER = {
    "ColliderURL": ["Models", "Models Raw"],
    "DiffuseURL": ["Images", "Images Raw"],
    "AssetbundleURL": ["Assetbundles"],
    "Nickname": [],
    "AssetbundleSecondaryURL": ["Assetbundles"],
    "ImageURL": ["Images", "Images Raw"],
    "MeshURL": ["Models", "Models Raw"],
    "SkyURL": ["Images", "Images Raw"],
    "BackURL": ["Images", "Images Raw"],
    "URL": ["Images", "Images Raw"],
    "FaceURL": ["Images", "Images Raw"],
    "ImageSecondaryURL": ["Images", "Images Raw"],
    "Item1": ["Audio"],
    "NormalURL": ["Images", "Images Raw"],
    "PDFUrl": ["PDF"],
}
FOLDER_EXT = {
    "Models": ".obj",
    "Assetbundles": ".unity3d",
    "PDF": ".pdf",
}


def get_mod_dir(json_file):
    while json_file.name != "Mods":
        json_file = json_file.parent
    return json_file


def copy_file(file, mod_dir, target_dir, target_name=None):
    target_file = target_dir / file.relative_to(mod_dir)
    if target_name is not None:
        target_file = target_file.with_stem(target_name)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(file, target_file)
    print(f"Copied {target_file.relative_to(target_dir)}")


def get_all_urls(data):
    urls = {}
    if not isinstance(data, dict):
        return urls
    for key, value in data.items():
        if isinstance(value, dict):
            urls.update(get_all_urls(value))
        elif isinstance(value, list):
            for item in value:
                urls.update(get_all_urls(item))
        elif isinstance(value, str) and value.startswith("http"):
            urls[value] = key
    return urls


def download_file(url, target_dir, file_stem):
    try:
        # 发送 HTTP GET 请求下载文件
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功

        # # 获取内容类型并确定文件扩展名
        folder = target_dir.name
        first_2048_bytes = response.raw.read(2048)
        if folder in FOLDER_EXT:
            ext = FOLDER_EXT[folder]
        else:
            #     content_type = response.headers.get("content-type")
            #     ext = mimetypes.guess_extension(content_type)
            #     if ext is None:
            #         ext = ".bin"  # 默认扩展名
            # 使用前 2048 字节来检测文件类型
            mime = magic.Magic(mime=True)
            content_type = mime.from_buffer(first_2048_bytes)
            ext = mimetypes.guess_extension(content_type)
        if ext is None:
            ext = ".bin"  # 默认扩展名

        # 确定文件名
        file_name = file_stem + ext
        file_path = target_dir / file_name
        target_dir.mkdir(parents=True, exist_ok=True)

        # 保存文件
        with open(file_path, "wb") as file:
            file.write(first_2048_bytes)
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"Downloaded {url}")
    except Exception as e:
        print(f"\033[91mFailed to download {url}\033[0m")


def process_url(url, key, mod_dir, target_dir):
    if not key in URL_FOLDER:
        print(f"Key {key} not found in URL_FOLDER")
        print(f"URL: {url}")
        return

    folder = URL_FOLDER[key]
    if len(folder) == 0:
        return
    folder = folder[0]  # the Raw folder is not considered
    file_stem_old = None
    if url.startswith("http://cloud-3.steamusercontent.com/"):
        file_stem_old = re.sub(r"[^a-zA-Z0-9]", "", url)
        url = url.replace(
            "http://cloud-3.steamusercontent.com/",
            "https://steamusercontent-a.akamaihd.net/",
        )
    if url.startswith("https://cloud-3.steamusercontent.com/"):
        file_stem_old = re.sub(r"[^a-zA-Z0-9]", "", url)
        url = url.replace(
            "https://cloud-3.steamusercontent.com/",
            "https://steamusercontent-a.akamaihd.net/",
        )
    file_stem = re.sub(r"[^a-zA-Z0-9]", "", url)

    files = (mod_dir / folder).glob(f"{file_stem}.*")
    files = list(files)
    if file_stem_old is not None:
        files_old = (mod_dir / folder).glob(f"{file_stem_old}.*")
        files_old = list(files_old)
        files += files_old

    if len(files) == 0:  # file not found
        # download the file
        download_file(url, target_dir / folder, file_stem)
    else:
        if len(files) > 1:
            print(f"\033[94mMultiple files found for {url}\033[0m")
            print(f"\033[94mFiles: {files}\033[0m")
        file_path = list(files)[0]
        copy_file(file_path, mod_dir, target_dir, file_stem)


def sanitize_folder_name(folder_name):
    # 移除不允许的字符（假设不允许的字符为：<>:"/\|?*）
    sanitized_name = re.sub(r'[<>:"/\\|?*]', "", folder_name)
    # 将连续的空格替换为单个空格
    sanitized_name = re.sub(r"\s+", " ", sanitized_name)
    return sanitized_name


def main():
    json_file = Path(sys.argv[1])
    mod_dir = get_mod_dir(json_file)

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_dir = Path(__file__).parent / sanitize_folder_name(data["SaveName"])
    thumbnail = json_file.parent / (json_file.stem + ".png")
    copy_file(thumbnail, mod_dir, target_dir)
    copy_file(json_file, mod_dir, target_dir)
    urls = get_all_urls(data)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_url, url, key, mod_dir, target_dir)
            for url, key in urls.items()
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
