#date: 2022-03-16T16:56:57Z
#url: https://api.github.com/gists/b9932a1bbe1c94ee3c38d73f8d76bc76
#owner: https://api.github.com/users/Snork2

import sys
from enum import Enum
from requests import request

class OrderBy(Enum):
    LIKES = "likes_count"
    VIEWS = "views_count"
    NEW = "newest"
    OLD = "oldest"
    POPULAR = "newest_popular"

per_page = 25
URL = f"https://coub.com/api/v2/timeline/likes?all=true&per_page={per_page}&order_by={OrderBy.NEW}&page="

def extract_coubs_links(content):
    with open("urls.txt", "a", encoding="utf-8") as urls:
        for coub in content["coubs"]:
            print("https://coub.com/view/{0}".format(coub["permalink"]), file=urls)

def do_request(page_number, token):
    while True:
        try:
            r = request("GET", f"{URL}{page_number}", cookies={"remember_token": token}, timeout=5)
            break
        except:
            print("Connection timeout, retry")
    return r.json()

# Press the green button in the gutter to run the script.
def get_likes():
    # Get token
    token = sys.argv[1]
    # Clear file
    file = open("urls.txt", "w", encoding="utf-8")
    file.close

    print("Page 1...")
    content = do_request(1, token)
    total_pages_ = content["total_pages"]

    extract_coubs_links(content)
    for page_num in range(2, total_pages_):
        print("Page {0}...".format(page_num))
        content = do_request(page_num, token)
        extract_coubs_links(content)

if __name__ == '__main__':
    get_likes()
