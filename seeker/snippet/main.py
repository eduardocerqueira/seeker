#date: 2024-03-08T17:00:19Z
#url: https://api.github.com/gists/7feb726ca7216ee1731b9fe0956ec189
#owner: https://api.github.com/users/Impre-visible

from pathlib import Path
from chocolate_app.plugins_loader import events
from chocolate_app.tables import Movies
from chocolate_app import DB

import threading
import requests

TPDB_API_KEY = "YOUR_FANART_TV_API_KEY"

@events.on(events.BEFORE_START)
def before_start():
    threading.Thread(target=generate_all_movies).start()

def generate_all_movies():
    from chocolate_app import app
    with app.app_context():
        movies = Movies.query.all()
        for movie in movies:
            if "fanart" in movie.cover:
                continue
            generate_for_movie(movie.id)

@events.on(events.NEW_MOVIE)
def new_movie(movie: Movies):
    threading.Thread(target=generate_for_movie, args=(movie.id)).start()

def generate_for_movie(movie_id: int):
    movie = Movies.query.filter_by(id=movie_id).first()
    cover_path = Path(movie.cover).parent
    banner_path = Path(movie.banner).parent
    file_name_cover = Path(movie.cover).name
    file_name_banner = Path(movie.banner).name

    if "http" in file_name_cover or "http" in file_name_banner:
        return

    new_file_name_cover = f"fanart_{file_name_cover}"
    new_file_name_banner = f"fanart_{file_name_banner}"

    movie_url = f"http://webservice.fanart.tv/v3/movies/{movie_id}?api_key={TPDB_API_KEY}"
    response = requests.get(movie_url)
    data = response.json()

    if "movieposter" not in data:
        return

    movies_posters = data["movieposter"]

    movie_poster = movies_posters[0]["url"]

    if "moviebanner" not in data:
        if "moviebackground" not in data:
            return
        movies_banners = data["moviebackground"]
    else:
        movies_banners = data["moviebanner"]

    movie_banner = movies_banners[0]["url"]

    poster_response = requests.get(movie_poster)
    banner_response = requests.get(movie_banner)

    with open(f"{cover_path}/{new_file_name_cover}", "wb") as f:
        f.write(poster_response.content)

    with open(f"{banner_path}/{new_file_name_banner}", "wb") as f:
        f.write(banner_response.content)

    movie.cover = f"{cover_path}/{new_file_name_cover}"
    movie.banner = f"{banner_path}/{new_file_name_banner}"

    DB.session.commit()