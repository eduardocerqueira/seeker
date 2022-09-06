#date: 2022-09-06T16:52:08Z
#url: https://api.github.com/gists/ab82c4e77a94bf13fec7f72703095991
#owner: https://api.github.com/users/mturoci

# Imports that we will eventually need.
import collections
from asyncio import ensure_future, gather, get_event_loop

from h2o_wave import data, site, ui
from httpx import AsyncClient

# Register page at "/" route.
page = site['/']

# Setup layout.
page['meta'] = ui.meta_card(
    box='',
    title='Wave comparison',
    layouts=[
        ui.layout(breakpoint='xs', zones=[
            ui.zone(name='header'),
            ui.zone(name='intro', direction=ui.ZoneDirection.ROW, size='500px'),
            ui.zone(name='plots1', direction=ui.ZoneDirection.ROW, size='300px'),
            ui.zone(name='plots2', direction=ui.ZoneDirection.ROW, size='300px'),
    ]),
])

# Render header.
page['header'] = ui.header_card(
    box='header',
    title='Wave competition comparison',
    subtitle="Let's see how well Wave does against its rivals.",
    image='https://wave.h2o.ai/img/h2o-logo.svg',
)

page.save()