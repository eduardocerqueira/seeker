#date: 2022-01-07T16:56:09Z
#url: https://api.github.com/gists/4f71daba8e376eecededa646b77aa505
#owner: https://api.github.com/users/jaymody

import requests
from bs4 import BeautifulSoup

base_url = "https://en.wikipedia.org/wiki/"


def page_to_url(page):
    return base_url + page


def url_to_page(url):
    return url.split("/")[-1]


def assert_url_exists(url):
    r = requests.get(url)
    if r.status_code != 200:
        raise ValueError(
            f"URL {r.url} could not be found: {r.status_code}, {r.reason}, {r.text}"
        )


def get_linked_pages(url):
    # TODO: add option to remove Bibliography, Further reading, External links, Disambiguation header section, etc ...
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    pages = []
    for link in soup.find("div", {"id": "bodyContent"}).find_all("a"):
        path = link.get("href")
        if not path:
            continue
        prefix, title = path[:6], path[6:]
        if prefix == "/wiki/" and ":" not in title:
            title = title.split("#")[0]
            pages.append(title)

    pages = list(set(pages))
    return pages

  
def bfs(src_url, trg_url, max_queue_size=1e6):
    assert src_url != trg_url
    assert_url_exists(src_url)
    assert_url_exists(trg_url)

    src_page = url_to_page(src_url)
    trg_page = url_to_page(trg_url)

    i = 0
    history = set([src_page])
    queue = [[src_page]]
    while queue:
        path = queue.pop(0)
        current_page = path[-1]
        current_url = page_to_url(current_page)
        linked_pages = get_linked_pages(current_url)
        if trg_page in linked_pages:
            return path + [trg_page]
        print(
            f"n = {i}",
            f"num_links = {len(linked_pages)}",
            f"queue_size = {len(queue)}",
            f"url = {current_url}",
            sep=", "
        )
        for linked_page in linked_pages:
            if linked_page in history:
                continue
            else:
                queue.append(path + [linked_page])
                if len(queue) > max_queue_size:
                    raise ValueError("Exceeded max_queue_size")
                history.add(linked_page)
        i += 1
    return False  


if __name__ == "__main__":
    print(bfs("https://en.wikipedia.org/wiki/Led_Zeppelin", "https://en.wikipedia.org/wiki/World_War_I"))