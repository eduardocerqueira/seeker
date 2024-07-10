#date: 2024-07-10T16:53:49Z
#url: https://api.github.com/gists/71ba97cd59962409e1eb4d5dd0f3e208
#owner: https://api.github.com/users/Nyashyker

from pprint import pprint
from pathlib import Path

# https://python-telegraph.readthedocs.io/en/latest/telegraph.html
import telegraph


def setContent(pictures: list) -> list:
    content: list = []
    for pic in pictures:
        content.append(
            {
                "tag": "figure",
                "children": [
                    {"tag": "img", "attrs": {"src": pic}},
                    {"tag": "figcaption", "children": [""]},
                ],
            }
        )
    return content


def uploadPictures(folder: Path) -> list[str]:
    """
    Працює тільки з .jpg
    """

    pictures: list = []
    for img in folder.glob("*.jpg"):
        pictures.append(TG.upload_file(img)[0]["src"])

    return pictures


TG = telegraph.api.Telegraph(
    "5fc3709d100f62b276ec573a389a11ea267692f67f6999e4568426a9105d"
)
# TG.create_account("DP", "Даєш переклад!")
# print(TG.get_access_token(), end= "**********"

path = Path
pictures: list[str] = uploadPictures()

# result = TG.edit_page(
result = TG.create_page(
    # path="",
    title="TestTestTest",
    content=setContent(pictures),
    author_name="",
    author_url="",
)
pprint(result)
