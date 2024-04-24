#date: 2024-04-24T17:02:59Z
#url: https://api.github.com/gists/0b996e4283b6718ed711e5b9457c5a43
#owner: https://api.github.com/users/jokereven

import asyncio
from twscrape import API, gather
from twscrape.logger import set_log_level
from datetime import datetime, timedelta
import requests
import json
import time
import pytz
import telebot

# -4183911082

bot = "**********"
bot.set_webhook()

def tg_bark(data):
    bot.send_message(ID, data, parse_mode='html',disable_web_page_preview=True)

async def get_user_id():
    api = API()  # or API("path-to.db") - default is `accounts.db`

    list = []
    user_list = ["xiaoshengfa123", "HHHHHAO11", "tugougan", "alphaFilterSB"]

    for user in user_list:
        user = await api.user_by_login(user)
        user_id = user.id
        list.append(user_id)

    return list

async def get_history():
    with open("tweet_ids.txt", "r") as file:
        lines = file.readlines()
        return lines

async def find(user_id):
    api = API()  # or API("path-to.db") - default is `accounts.db`

    # 获取到用户最近的数据,根据时间如果是最近的就推送
    ut = await gather(api.user_tweets(user_id, limit=20))  # list[Tweet]
    utar = await gather(api.user_tweets_and_replies(user_id, limit=20))  # list[Tweet]

    # lt = await gather(api.liked_tweets(user_id, limit=20))  # list[Tweet]

    current_time = datetime.now(pytz.utc)

    for tweet in ut:
        tweet_date = tweet.date

        time_difference = current_time - tweet_date

        if time_difference <= timedelta(minutes=5):

            id = tweet.id
            # id 存贮到文件

            if str(id) in await get_history():
                continue
            else:
                username = tweet.user.username
                url = tweet.url
                rawDescription = tweet.user.rawDescription
                tg_bark(f"🐦 <a href='https://twitter.com/{username}'>{username}</a> 发布了新推文\n\n{rawDescription}\n\n<a href='{url}'>查看推文</a>")

            # 将 ID 写入文件
                with open("tweet_ids.txt", "a") as file:
                    file.write(str(id) + "\n")

async def x():
    api = API()  # or API("path-to.db") - default is `accounts.db`

    await api.pool.add_account("", "", "", "")
    await api.pool.login_all()

    get_id = await get_user_id()

    for user_id in get_id:
        await find(user_id)

async def main():
    while True:
        await x()
        await asyncio.sleep(30)  # 每 30 秒执行一次 find

if __name__ == "__main__":
    asyncio.run(main()).run(main())