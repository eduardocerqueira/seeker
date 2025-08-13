#date: 2025-08-13T17:11:04Z
#url: https://api.github.com/gists/45958eef47d0b600a1d907e73c87a750
#owner: https://api.github.com/users/csuhan

import sys
import json
import random
import time
import os
import tqdm
import curl_cffi.requests as curl_requests

sys.setrecursionlimit(10000)

SORT = "Newest"
PERIOD = "AllTime"
LIMIT = 200
URL = "https://civitai.com/api/v1/images?limit={limit}&period={period}&sort={sort}"

# 保存目录
SAVE_DIR = "civitai_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# resume 文件记录已抓取的 nextCursor
RESUME_FILE = os.path.join(SAVE_DIR, "resume.json")

# 尝试读取上次抓取状态
if os.path.exists(RESUME_FILE):
    with open(RESUME_FILE, "r", encoding="utf-8") as f:
        resume_data = json.load(f)
        next_cursor = resume_data.get("nextCursor")
        image_ids = set(resume_data.get("image_ids", []))
else:
    next_cursor = None
    image_ids = set()

pbar = tqdm.tqdm()


def save_resume(next_cursor, image_ids):
    with open(RESUME_FILE, "w", encoding="utf-8") as f:
        json.dump({"nextCursor": next_cursor, "image_ids": list(image_ids)}, f, ensure_ascii=False, indent=2)


def get_images(
    progress_bar: tqdm.tqdm,
    ids_set: set,
    url: str = None,
):
    if url is None:
        url = URL.format(limit=LIMIT, period=PERIOD, sort=SORT)
        if next_cursor:
            url += f"&cursor={next_cursor}"
        progress_bar.close()
        progress_bar = tqdm.tqdm()

    resp = curl_requests.get(
        url,
        impersonate="chrome",
        timeout=10,
    )
    data = resp.json()

    if "error" in data:
        print(data)
        return

    next_page = data["metadata"].get("nextPage", None)
    next_cursor_local = data["metadata"].get("nextCursor", None)
    items = data["items"]

    # 过滤重复 ID
    new_items = []
    for item in items:
        if item["id"] in ids_set:
            continue
        meta = item.get("meta", {})
        seed = None
        if meta:
            seed = meta.get("seed", None)
            
            _ = meta.pop("comfy", None)
        if seed:
            item["meta"]["seed"] = str(seed)

        ids_set.add(item["id"])
        new_items.append(item)

    # 保存当前页
    if new_items:
        page_index = len(os.listdir(SAVE_DIR))
        page_file = os.path.join(SAVE_DIR, f"page_{page_index:04d}.json")
        with open(page_file, "w", encoding="utf-8") as f:
            json.dump(new_items, f, ensure_ascii=False, indent=2)

    progress_bar.set_postfix({"nextCursor": str(next_cursor_local), "count": len(ids_set)})
    progress_bar.update()

    # 保存 resume 文件
    save_resume(next_cursor_local, ids_set)

    if next_page is not None:
        time.sleep(random.uniform(0.5, 1.5))
        get_images(progress_bar, ids_set, next_page)


get_images(progress_bar=pbar, ids_set=image_ids, url=None)
