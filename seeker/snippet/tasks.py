#date: 2022-07-14T16:52:08Z
#url: https://api.github.com/gists/13627b120e9503254fce01f7ce18a61b
#owner: https://api.github.com/users/2rist116

from celery import Celery
from webapp.ya_music.ya_music import create_new_playlist
from yandex_music import Client
from flask_login import current_user

celery_app = Celery('tasks', broker='redis://redis:6379/0')


@celery_app.task()
def new_playlist(playlist_ids, client):
    token = current_user.yandex_token
    client = Client(token).init()
    create_new_playlist(playlist_ids, client)