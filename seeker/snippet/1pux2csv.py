#date: 2021-12-09T17:10:13Z
#url: https://api.github.com/gists/a0f6d92d41c081eddf874a05f670d540
#owner: https://api.github.com/users/tyhenry

import csv
import json
import os

def convert(file_in, dir_out=""):
    with open(file_in, 'r', encoding='utf8') as json_file:
        data = json.load(json_file)
        for account in data["accounts"]:
            print(f"Processing account: {account['attrs']['name']}")
            for vault in account["vaults"]:
                vault_name = vault["attrs"]["name"]
                print(f"Processing vault: {vault_name}")
                csv_dir = dir_out if dir_out != "" else os.path.dirname(file_in)
                with open(os.path.join(csv_dir, f"{vault_name}.csv"), "w", newline='', encoding='utf8') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(["Title", "URL", "Username", "Password"])
                    for item in vault["items"]:
                        item = item["item"]
                        overview = item["overview"]
                        title = overview["title"]
                        url = overview["url"]
                        print(f"Processing item: {title}")
                        username, password = None, None
                        for field in item["details"]["loginFields"]:
                            if "designation" not in field:
                                continue
                            if field["designation"] == "username":
                                username = field["value"]
                            if field["designation"] == "password":
                                password = field["value"]
                        writer.writerow([title, url, username, password])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert from 1pux data format to csv")
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_dir', nargs='?', default="", type=str)
    args = parser.parse_args()
    convert(args.input_file, args.output_dir)