#date: 2025-10-16T17:09:30Z
#url: https://api.github.com/gists/bd434bb4eeaa9564bf108632d5040821
#owner: https://api.github.com/users/psychoticbeef

import os
import json
import time
import pathlib
import re
import sys
import requests
from markdownify import markdownify as md_convert

API_BASE = "https://api.kanka.io/1.0"
STATE_FILE = "seen_entities.json"
OUT_DIR = pathlib.Path("kanka_downloads")

# Map Kanka entity "type" to the plural endpoint name for the child resource
TYPE_TO_ENDPOINT = {
    "character": "characters",
    "location": "locations",
    "family": "families",
    "organisation": "organisations",  # Kanka uses British spelling
    "item": "items",
    "note": "notes",
    "quest": "quests",
    "journal": "journals",
    "race": "races",
    "event": "events",
    "ability": "abilities",
    "calendar": "calendars",
    "map": "maps",
    "tag": "tags",
    # Add more as needed
}

def get_session(token: "**********":
    s = requests.Session()
    s.headers.update({
        "Authorization": "**********"
        "Accept": "application/json",
        "Content-type": "application/json",
        "User-Agent": "kanka-downloader/0.1",
    })
    return s

def list_campaigns(session: requests.Session):
    r = session.get(f"{API_BASE}/campaigns")
    r.raise_for_status()
    return r.json().get("data", [])

def iter_entities(session: requests.Session, campaign_id: int):
    page = 1
    while True:
        r = session.get(f"{API_BASE}/campaigns/{campaign_id}/entities", params={"page": page})
        r.raise_for_status()
        payload = r.json()
        data = payload.get("data", [])
        for ent in data:
            yield ent
        links = payload.get("links") or {}
        if not links.get("next"):
            break
        page += 1

def sanitize_filename(name: str) -> str:
    name = name.strip().replace("/", "-").replace("\\", "-")
    return re.sub(r"[^A-Za-z0-9 _.-]", "_", name)

def load_seen_ids() -> set:
    if not os.path.exists(STATE_FILE):
        return set()
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        try:
            arr = json.load(f)
            return set(arr)
        except Exception:
            return set()

def save_seen_ids(seen: set):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(seen), f, indent=2)

def get_entity_details(session: requests.Session, campaign_id: int, entity_id: int) -> dict:
    # Fetch the specific entity to get its type and child_id
    r = session.get(f"{API_BASE}/campaigns/{campaign_id}/entities/{entity_id}")
    r.raise_for_status()
    return r.json().get("data", {})

def get_child_entry(session: requests.Session, campaign_id: int, etype: str, child_id: int) -> dict:
    endpoint = TYPE_TO_ENDPOINT.get(etype)
    if not endpoint:
        raise ValueError(f"Unsupported entity type '{etype}'. Add it to TYPE_TO_ENDPOINT.")
    r = session.get(f"{API_BASE}/campaigns/{campaign_id}/{endpoint}/{child_id}")
    r.raise_for_status()
    return r.json().get("data", {})

def download_new_entries(session: requests.Session, campaign_id: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seen = load_seen_ids()
    new_count = 0
    max_retries = 10
    retry_wait_seconds = 60

    for ent in iter_entities(session, campaign_id):
        eid = ent.get("id")
        if eid in seen:
            continue

        try:
            # Wrap entity-specific calls so we can retry on 429 (rate limit)
            attempt = 0
            while True:
                try:
                    details = get_entity_details(session, campaign_id, eid)
                    break
                except requests.HTTPError as he:
                    status = he.response.status_code if he.response is not None else None
                    if status == 429 and attempt < max_retries:
                        attempt += 1
                        print(f"Rate limited (429) when fetching details for entity {eid}."
                              f" Waiting {retry_wait_seconds}s and retrying (attempt {attempt}/{max_retries})...")
                        time.sleep(retry_wait_seconds)
                        continue
                    raise
            etype = details.get("type")
            child_id = details.get("child_id")
            name = details.get("name") or f"entity_{eid}"

            if not etype or child_id is None:
                print(f"Skipping entity {eid} (missing type/child_id)")
                seen.add(eid)
                continue

            # Fetch child entry with same retry-on-429 behavior
            attempt = 0
            while True:
                try:
                    child = get_child_entry(session, campaign_id, etype, child_id)
                    break
                except requests.HTTPError as he:
                    status = he.response.status_code if he.response is not None else None
                    if status == 429 and attempt < max_retries:
                        attempt += 1
                        print(f"Rate limited (429) when fetching child entry for entity {eid}."
                              f" Waiting {retry_wait_seconds}s and retrying (attempt {attempt}/{max_retries})...")
                        time.sleep(retry_wait_seconds)
                        continue
                    raise
            entry_html = child.get("entry") or ""
            # Convert HTML to Markdown if possible, otherwise keep raw HTML.
            entry_md = None
            if md_convert:
                try:
                    entry_md = md_convert(entry_html)
                except Exception as e:
                    print(f"markdownify conversion failed for entity {eid}: {e}")
                    entry_md = None

            if entry_md is None and _h2t is not None:
                try:
                    entry_md = _h2t.handle(entry_html)
                except Exception as e:
                    print(f"html2text conversion failed for entity {eid}: {e}")
                    entry_md = None

            # If both converters unavailable/failed, fall back to raw HTML but save with .md
            if entry_md is None:
                print("Warning: no HTML->Markdown converter available; saving raw HTML into .md file."
                      " Install 'markdownify' or 'html2text' for better results.")
                entry_md = entry_html

            subdir = OUT_DIR / etype
            subdir.mkdir(parents=True, exist_ok=True)
            fname = f"{eid} - {sanitize_filename(name)}.md"
            (subdir / fname).write_text(entry_md, encoding="utf-8")
            print(f"Saved {etype} #{child_id} ({name}) -> {subdir / fname}")

            seen.add(eid)
            new_count += 1

            # Be gentle; adjust if you hit rate limits.
            time.sleep(0.1)

        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            print(f"HTTP error on entity {eid}: {status} {e}")
            # If unauthorized or forbidden, you likely lack access to the campaign or a private entity.
            seen.add(eid)  # avoid retry loop; remove this if you want to retry later

        except Exception as e:
            print(f"Error on entity {eid}: {e}")

    save_seen_ids(seen)
    print(f"Done. New entries saved: {new_count}")

def main():
    token = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
        print("Please set KANKA_TOKEN environment variable.", file= "**********"
        sys.exit(1)

    session = "**********"
    campaigns = list_campaigns(session)
    if not campaigns:
        print("No campaigns found or token lacks access.")
        return

    # Print campaigns and pick one
    print("Your campaigns:")
    for c in campaigns:
        print(f"- {c.get('id')}: {c.get('name')}")

    raw = input("Enter campaign ID to use (or leave blank to abort): ").strip()
    if not raw:
        return
    campaign_id = int(raw)

    download_new_entries(session, campaign_id)

if __name__ == "__main__":
    main()