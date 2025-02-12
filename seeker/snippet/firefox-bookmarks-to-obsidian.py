#date: 2025-02-12T16:42:58Z
#url: https://api.github.com/gists/5517b73a7f74e8cbff65ab76983e684c
#owner: https://api.github.com/users/Lu1sDV

#!/usr/bin/env python3
import argparse
import os
import configparser
import sqlite3

OBSIDIAN_VAULT_PATH = ""
FIREFOX_PROFILE_PATH = ""

try:
    SYSTEM_MAX_PATH_LENGTH = os.pathconf('/', 'PC_PATH_MAX')
    SYSTEM_MAX_FILENAME_LENGTH = os.pathconf('/', 'PC_NAME_MAX')
except OSError as e:
    print(e)
    exit(1)

def load_firefox_database(db_path):
    """Loads bookmarks from a Firefox database file using optimized SQL.
        Args:
            path (str): The path to the Firefox database file (places.sqlite).
        Returns:
            list: A list of bookmarks, where each bookmark is a list of tags
                  (strings) representing the folder hierarchy, and a tuple
                  containing the bookmark title and URL.  Returns empty list if
                  there are any exceptions.
                  Example:
                  [['tag1', 'tag2', ('bookmark_title', 'bookmark_url')]]
    """
    books = list()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()

    try:
        res = cur.execute("""
            SELECT
                p.url,
                bm.title AS bookmark_title,
                GROUP_CONCAT(DISTINCT tag_bm.title) AS tags,
                bm.parent
            FROM
                moz_bookmarks bm
            JOIN
                moz_places p ON bm.fk = p.id
            LEFT JOIN
                moz_bookmarks tag_bm ON bm.parent = tag_bm.id
            WHERE
                bm.type = 1
            GROUP BY
                p.url, bookmark_title, bm.parent
        """)

        for row in res.fetchall():
            url, bookmark_title, tags_str, parent_id = row
            bookmark_tags = []

            # Extract tags and handle hierarchy (simplified, may need further refinement for complex hierarchies)
            if tags_str:
                bookmark_tags = [tag.strip() for tag in tags_str.split(',')]

            # Fetch parent folder titles - simplified, fetches only immediate parent
            parent_tags = []
            current_parent_id = parent_id
            while current_parent_id and current_parent_id != 2 and current_parent_id != 1: # Stop at "Bookmarks Menu" (id 2) and "Bookmarks Toolbar" (id 1)
                parent_res = cur.execute('SELECT title, parent FROM moz_bookmarks WHERE id=?', (current_parent_id,))
                parent_row = parent_res.fetchone()
                if parent_row:
                    parent_title, current_parent_id = parent_row
                    parent_tags.append(parent_title)
                else:
                    break # No more parent found

            bookmark_tags.extend(parent_tags)


            bookmark_title = bookmark_title or ''
            bookmark_tags = bookmark_tags[::-1]  # Reverse to get correct hierarchy order
            bookmark_tags = [tag for tag in bookmark_tags if tag not in ("Bookmarks Menu", "Bookmarks Toolbar")] # Remove Bookmarks Menu and Toolbar
            bookmark_tags.append(tuple((bookmark_title, url)))
            books.append(bookmark_tags)


    except Exception as e:
        print(f"Database error: {e}")
        books = [] 
    finally:
        cur.close()
        conn.close()
    return books
   

def get_default_firefox_profile_path():
    """
    Finds the default Firefox profile path from the profile.ini file.

    Returns:
        str: Path to the default Firefox profile, or None if not found.
    """
    mozilla_dir = os.path.expanduser("~/.mozilla/firefox")
    profile_ini_path = os.path.join(mozilla_dir, "profiles.ini")

    if not os.path.exists(profile_ini_path):
        return None

    config = configparser.ConfigParser()
    config.read(profile_ini_path)

    for section in config.sections():
        if section.startswith("Install"):
            if "Default" in config[section]:
                default_profile_name = config[section]["Default"]
                profile_path_relative =  default_profile_name
                return os.path.join(mozilla_dir, profile_path_relative)

    return None

def copy_bookmarks_to_obsidian(bookmarks, obsidian_vault_path):
    """
    Copies the Firefox bookmarks file to the Obsidian vault.

    Args:
        bookmarks (list): List of bookmarks to copy.
        obsidian_vault_path (str): Path to the Obsidian vault directory.
    Returns:
        bool: True if the copy was successful, False otherwise.
    """
    
    dirs = list()
    for i, b in enumerate(bookmarks):
        if len(b) == 1: 
            b.insert(0, "Orphaned")
        dirs.append("/".join(b[:-1]))
    
    dirs = list(set(dirs))

    [os.makedirs(os.path.join(obsidian_vault_path, d), exist_ok=True) for d in dirs]

    for b in bookmarks:

        bookmark_title, bookmark_url = b[-1]
        bookmark_title = bookmark_title.replace('/', "-") # sanitize the title to avoid creating subfolders
        bookmark_title = f"{bookmark_title}.md"
        bookmark_tags = "/".join(b[:-1])
        
        bookmark_path = os.path.join(obsidian_vault_path, bookmark_tags, bookmark_title)

        if len(bookmark_path) > SYSTEM_MAX_PATH_LENGTH:
            print(f"Error: Path too long: {bookmark_path}")
            continue
        
        if len(bookmark_title) > SYSTEM_MAX_FILENAME_LENGTH:
            print(f"Error: Filename too long: {bookmark_title}")
            continue

        with open(bookmark_path, "w") as f:
            f.write(f"[{bookmark_title}]({bookmark_url})")

    return True
    
def main():
    parser = argparse.ArgumentParser(description="Copy Firefox bookmarks to Obsidian Vault.")

    parser.add_argument(
        "--firefox-profile",
        type=str,
        default=False if FIREFOX_PROFILE_PATH == "" else FIREFOX_PROFILE_PATH,
        help="Path to the Firefox profile directory. If not provided, tries to find the default.",
    )
    parser.add_argument(
        "--obsidian-vault",
        type=str,
        default=False if OBSIDIAN_VAULT_PATH == "" else OBSIDIAN_VAULT_PATH,
        help="Path to the Obsidian Vault directory target folder.",
        required=True if OBSIDIAN_VAULT_PATH == "" else False,
    )

    args = parser.parse_args()

    obsidian_vault_path = args.obsidian_vault
    firefox_profile_path = args.firefox_profile

    if not args.firefox_profile:
        print("Firefox profile path not provided. Trying to find default profile...")
        firefox_profile_path = get_default_firefox_profile_path()
        if not firefox_profile_path:
            print("Error: Could not find default Firefox profile. Please provide the Firefox profile path using --firefox-profile argument.")
            return

    print(f"Firefox profile path: {firefox_profile_path}")
    if not os.path.exists(firefox_profile_path):
        print(f"Error: Firefox profile path not found: {firefox_profile_path}")
        return

    if not os.path.exists(obsidian_vault_path):
        print(f"Error: Obsidian vault path not found: {obsidian_vault_path}")
        return

    bookmarks = load_firefox_database(os.path.join(firefox_profile_path, "places.sqlite"))

    if copy_bookmarks_to_obsidian(bookmarks, obsidian_vault_path):
        print("Script finished successfully.")
    else:
        print("Script finished with errors.")


if __name__ == "__main__":
    main()