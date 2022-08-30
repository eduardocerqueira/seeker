#date: 2022-08-30T17:08:29Z
#url: https://api.github.com/gists/549be85bb5de2d98025fa7178d48b62d
#owner: https://api.github.com/users/gaycookie

from threading import Timer
from requests import request
import sqlite3

api_url = "https://socialclub.rockstargames.com/events/eventlisting?pageId=1&gameId=GTAV"
article_url = "https://socialclub.rockstargames.com/events/{}/{}"
webhook_url = ""

def init_database():
  db_conn = sqlite3.connect("database.sqlite")
  cursor = db_conn.cursor()
  cursor.execute("CREATE TABLE IF NOT EXISTS articles (hash varchar(16) PRIMARY KEY, slug text);")
  db_conn.close()

def main():
  db_conn = sqlite3.connect("database.sqlite")
  req = request("get", api_url)
  data = req.json()
  
  for article in data["events"]:
    if article["isLive"] == True:
      cursor = db_conn.cursor()
      cursor.execute("SELECT * FROM articles WHERE hash = ?;", (article["urlHash"], ))
      stored = cursor.fetchone()

      if not stored:
        cursor = db_conn.cursor()
        cursor.execute("INSERT INTO articles (hash, slug) VALUES (?, ?);", (article["urlHash"], article["slug"]))
        db_conn.commit()

        request("post", webhook_url, json={
          "username": "Newswire",
          "avatar_url": "https://editors.dexerto.com/wp-content/uploads/2021/11/17/GTA-Vice-City-Rockstar-Logo.jpg",
          "content": "[{}]({})".format(article["headerText"], article_url.format(article["urlHash"], article["slug"]))
        })

        print(article["headerText"])

  db_conn.close()
  Timer(60 * 30, main).start()

if __name__ == "__main__":
  init_database()
  main()