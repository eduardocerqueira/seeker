#date: 2022-01-19T17:04:24Z
#url: https://api.github.com/gists/6b2afd81cfc15ad4084d3da620bef73f
#owner: https://api.github.com/users/eliasdorneles

from fastapi import FastAPI, APIRouter, Request
from fastapi.responses import RedirectResponse, HTMLResponse


router = APIRouter()

@router.get('/form')
def form():
    return HTMLResponse("""
    <html>
    <form action="/event/create" method="POST">
    <button>Send request</button>
    </form>
    </html>
    """)

@router.post('/create')
async def event_create(
        request: Request
):
    event = {"id": 123}
    redirect_url = request.url_for('get_event', **{'pk': event['id']})
    return RedirectResponse(redirect_url, status_code=303)


@router.get('/{pk}')
async def get_event(
        request: Request,
        pk: int,
):
    return f'<html>oi pk={pk}</html>'

app = FastAPI(title='Test API')

app.include_router(router, prefix="/event")

# run:
#
# uvicorn --reload --host 0.0.0.0 --port 3000 example:app
#
# then point your browser to: http://localhost:3000/event/create
