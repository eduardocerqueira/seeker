#date: 2022-01-05T16:57:29Z
#url: https://api.github.com/gists/6d0602bfa939a01844f645c608afb85a
#owner: https://api.github.com/users/vrslev

import os

import arel
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates("templates")

if _debug := os.getenv("DEBUG"):
    hot_reload = arel.HotReload(paths=[arel.Path(".")])
    app.add_websocket_route("/hot-reload", route=hot_reload, name="hot-reload")
    app.add_event_handler("startup", hot_reload.startup)
    app.add_event_handler("shutdown", hot_reload.shutdown)
    templates.env.globals["DEBUG"] = _debug
    templates.env.globals["hot_reload"] = hot_reload


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})
