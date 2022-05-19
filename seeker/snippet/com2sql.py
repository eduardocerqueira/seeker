#date: 2022-05-19T17:39:49Z
#url: https://api.github.com/gists/9a06ce1fb21bc26eb6d32c20782386dc
#owner: https://api.github.com/users/LunarWatcher

import sqlite3 as sl3
import re

con = sl3.connect("comments.db")

cur = con.cursor()

cur.execute("""CREATE TABLE IF NOT EXISTS Comments (UserID INTEGER, CommentID INTEGER PRIMARY KEY, Content TEXT, PostID INTEGER)""")
con.commit()

with open("Comments.xml") as f:
    print("File loaded")

    cnt = 0
    comments = 1

    while line := f.readline():
        if "<row" not in line:
            continue

        id = re.search("Id=\"(\\d+)\"", line)
        if (id is None):
            id = comments + 1
            comments += 1
        else:
            id = id.group(1)
            comments = id

        postId = re.search("PostId=\"(\\d+)\"", line)
        if (postId is not None):
            postId = postId.group(1)

        text = re.search("Text=\"(.*)\" CreationDate=", line)
        text = text.group(1)

        userId = re.search("UserId=\"(\\d+)\"", line)
        if (userId is None):
            userId = -1
        else:
            userId = userId.group(1)

        cur.execute("""INSERT OR REPLACE INTO Comments(UserID, CommentID, Content, PostID) VALUES (?, ?, ?, ?)""", (userId, id, text, postId))

        cnt += 1
        if (cnt % 10000 == 0):
            con.commit()
            cnt = 0

con.commit()
con.close()
