#date: 2025-05-05T16:53:03Z
#url: https://api.github.com/gists/e8e74e26e194efd0317879e30655cd07
#owner: https://api.github.com/users/stquinn

import argparse
import logging
from xml.etree import ElementTree
from plexapi.server import PlexServer
from jellyfin_apiclient_python import JellyfinClient

# ----------------- SETUP LOGGING ------------------
def setup_logging(to_stdout=False):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    if to_stdout:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        root.addHandler(ch)
    else:
        fh = logging.FileHandler("migration_log.txt")
        fh.setFormatter(formatter)
        root.addHandler(fh)

# ----------------- PLEX SECTION ------------------
 "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"p "**********"l "**********"e "**********"x "**********"_ "**********"i "**********"t "**********"e "**********"m "**********"s "**********"( "**********"p "**********"l "**********"e "**********"x "**********"_ "**********"u "**********"r "**********"l "**********", "**********"  "**********"p "**********"l "**********"e "**********"x "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********", "**********"  "**********"m "**********"a "**********"t "**********"c "**********"h "**********"_ "**********"b "**********"y "**********") "**********": "**********"
    plex = "**********"
    ids_to_labels = {}
    headers = {"X-Plex-Token": "**********"

    for section in plex.library.sections():
        if section.TYPE not in ["movie", "show"]:
            continue
        for item in section.all():
            label_tags = [label.tag for label in getattr(item, 'labels', [])]
            if not label_tags:
                continue
            rating_key = item.ratingKey
            matched_id = None
            guid = item.guid or ""

            if f"{match_by}://" in guid:
                matched_id = guid.split(f"{match_by}://")[1].split("?")[0]
            else:
                try:
                    meta_url = f"{plex_url}/library/metadata/{rating_key}"
                    import requests
                    response = requests.get(meta_url, headers=headers, timeout=5)
                    response.raise_for_status()
                    root = ElementTree.fromstring(response.content)
                    for g in root.findall(".//Guid"):
                        gid = g.attrib.get("id", "")
                        if gid.startswith(f"{match_by}://"):
                            matched_id = gid.split(f"{match_by}://")[1]
                            break
                except Exception as e:
                    logging.warning(f"Metadata fetch failed for {item.title}: {e}")
                    continue

            if matched_id:
                ids_to_labels[matched_id] = {
                    "title": item.title,
                    "labels": label_tags
                }
                logging.info(f"Matched {item.title} -> {match_by.upper()} ID {matched_id}")

    return ids_to_labels

# ----------------- JELLYFIN SECTION ------------------
 "**********"d "**********"e "**********"f "**********"  "**********"j "**********"e "**********"l "**********"l "**********"y "**********"f "**********"i "**********"n "**********"_ "**********"c "**********"o "**********"n "**********"n "**********"e "**********"c "**********"t "**********"( "**********"j "**********"e "**********"l "**********"l "**********"y "**********"f "**********"i "**********"n "**********"_ "**********"u "**********"r "**********"l "**********", "**********"  "**********"u "**********"s "**********"e "**********"r "**********"n "**********"a "**********"m "**********"e "**********", "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********") "**********": "**********"
    client = JellyfinClient()
    client.config.app("PlexToJellyfinSync", "1.0", "label-sync-client", "unique-id")
    client.config.data["auth.ssl"] = jellyfin_url.startswith("https")
    client.config.data["auth.serveraddress"] = jellyfin_url

    try:
        client.auth.connect_to_address(jellyfin_url)
        client.auth.login(jellyfin_url, username, password)
        return client
    except Exception as e:
        logging.error(f"❌ Failing server connection. ERROR msg: {e}")
        raise SystemExit(1)
    except Exception as e:
        logging.error(f"❌ Failing server connection. ERROR msg: {e}")
        raise SystemExit(1)

def find_jellyfin_item(client, match_by, external_id):
    try:
        results = client.jellyfin._get("Items", params={
            "Recursive": True,
            "IncludeItemTypes": "Movie,Series",
            "Fields": "ProviderIds",
            "api_key": "**********"
        })
        for item in results.get("Items", []):
            provider_ids = item.get("ProviderIds", {})
            if provider_ids.get(match_by.capitalize()) == external_id:
                item_type = item.get("Type")
                print(f"🧩 Found Jellyfin item match: {item['Name']} ({item_type}) ID={item['Id']}")
                if item_type not in ["Movie", "Series"]:
                    logging.warning(f"Skipping unsupported Jellyfin item type: {item_type} for {item['Name']}")
                    continue
                return item["Id"], item.get("Name")
    except Exception as e:
        logging.error(f"❌ Failed to find item in Jellyfin: {e}")
    return None, None

def apply_tags(client, item_id, new_tags, dry_run):
    try:
        metadata = client.jellyfin._get(f"Items/{item_id}")
    except Exception as e:
        logging.error(f"❌ Failed to fetch item metadata: {e}")
        return

    # Strip down metadata to only safe, updatable fields
    allowed_keys = [
        "Name", "OriginalTitle", "SortName", "ForcedSortName",
        "Tags", "Genres", "Studios", "ProviderIds",
        "PremiereDate", "ProductionYear", "OfficialRating",
        "Overview", "CommunityRating", "RunTimeTicks"
    ]
    update_payload = {k: v for k, v in metadata.items() if k in allowed_keys}

    existing_tags = metadata.get("Tags", [])
    update_payload["Tags"] = sorted(set(existing_tags + new_tags))

    if dry_run:
        logging.info(f"[DRY RUN] Would apply tags to {metadata.get('Name')} -> {update_payload['Tags']}")
        return

    try:
        client.jellyfin._post(f"Items/{item_id}", json=update_payload)
        logging.info(f"✅ Updated tags for {metadata.get('Name')}: {update_payload['Tags']}")
    except Exception as e:
        logging.error(f"❌ Failed to update tags: {e}")

# ----------------- MAIN SCRIPT ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plex-url', required=True)
    parser.add_argument('--plex-token', required= "**********"
    parser.add_argument('--jellyfin-url', required=True)
    parser.add_argument('--jellyfin-username', required=True)
    parser.add_argument('--jellyfin-password', required= "**********"
    parser.add_argument('--match-by', choices=['tvdb', 'imdb'], default='tvdb')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--log-to-stdout', action='store_true')
    args = parser.parse_args()

    setup_logging(to_stdout=args.log_to_stdout)

    client = "**********"
    ids_to_labels = "**********"

    total, tagged, skipped = len(ids_to_labels), 0, 0
    for external_id, data in ids_to_labels.items():
        title, labels = data['title'], data['labels']
        print(f"🔍 {title} ({args.match_by.upper()} ID {external_id})")
        item_id, name = find_jellyfin_item(client, args.match_by, external_id)
        if item_id:
            apply_tags(client, item_id, labels, args.dry_run)
            tagged += 1
        else:
            logging.warning(f"⚠️ No match in Jellyfin for {external_id} ({title})")
            skipped += 1

    print("\n=== SUMMARY ===")
    print(f"Matched Plex items: {total}")
    print(f"{'Would tag' if args.dry_run else 'Tagged'} in Jellyfin: {tagged}")
    print(f"Skipped or unmatched: {skipped}")

if __name__ == "__main__":
    main()
`