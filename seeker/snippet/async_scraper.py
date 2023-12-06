#date: 2023-12-06T17:10:35Z
#url: https://api.github.com/gists/b69c94f99cff13a2d2ee185de1e2e0ed
#owner: https://api.github.com/users/yudevan

import asyncio
import aiofiles
import aiohttp
import logging
import re
import sys
import os

import lxml.html

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

HEADERS = {
    'USER-AGENT': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.85 Safari/537.36'
}
MAX_PICS = 10
SUBREDDIT = 'earthporn'


class WorkerPool:
    def __init__(self, loop, coro, worker_count, options):
        self.loop = loop or asyncio.get_event_loop()
        self.result = None
        self.q = asyncio.Queue(loop=self.loop)
        self.coro = coro
        self.worker_count = worker_count
        self.options = options

    async def run(self):
        workers = [asyncio.Task(self.coro(self.loop, self.q, self.options))
                   for _ in range(self.worker_count)]
        await self.q.join()
        for w in workers:
            w.cancel()


async def _fetch(loop, q, options):
    # the loop makes sure the coroutine doesn't return and keeps going
    while True:
        try:
            ''' Pull a url from the queue and request it '''
            url, utype = await q.get()
            logger.debug('url: %s, url type: %s', url, utype)
            async with aiohttp.ClientSession(loop=loop, headers=HEADERS) as session:
                async with session.get(url) as resp:
                    if utype == 'seed':
                        text = await resp.text()
                        for link in _parse_links(text):
                            q.put_nowait((link, 'post'))
                    elif utype == 'post':
                        text = await resp.text()
                        q.put_nowait((_get_image_link(text), 'img'))
                    else:
                        outdir = os.path.join('/tmp', SUBREDDIT, options['mods'].replace('/', '_'))
                        await _get_image(resp, outdir)
            logger.debug('about to finish task, queue size: %s', q.qsize())
            q.task_done()
        except asyncio.CancelledError:
            break
        except:
            logger.exception('error')
            raise


def _parse_links(text):
    logger.debug('getting post links')
    html = lxml.html.fromstring(text)
    links = html.xpath("//a[contains(@class, title)]/@href")
    logger.debug(links)
    links = ['http://imgur.com' + link for link in links
             if re.search('^/r/{}/[a-zA-Z0-9]{{7}}'.format(SUBREDDIT), link, re.I)]
    logger.debug(links)
    return links[:MAX_PICS]


def _get_image_link(text):
    logger.debug('getting actual image link')
    html = lxml.html.fromstring(text)
    video_link = html.xpath('//div[@class="post-image"]//video/following-sibling::meta[@itemprop="contentURL"]/@content')
    if video_link:
        logger.debug('returning video link')
        return video_link[0]
    link = html.xpath('//div[@class="post-image"]//img/@src')[0]
    return link


async def _get_image(resp, outdir):
    logger.debug('saving image')
    fn = resp.url.split('/')[-1]

    async with aiofiles.open('{}/{}'.format(outdir, fn), 'wb') as f:
        data = await resp.read()
        await f.write(data)


def main(mods=''):
    try:
        outdir = os.path.join('/tmp', SUBREDDIT, mods.replace('/', '_'))
        os.makedirs(outdir)
    except:
        logger.warn("oops couldn't create folder")

    try:
        loop = asyncio.get_event_loop()
        wp = WorkerPool(loop, _fetch, 5, options={'sr': SUBREDDIT, 'mods': mods})
        seed_url = 'http://imgur.com/r/{}{}'.format(SUBREDDIT, mods)
        wp.q.put_nowait((seed_url, 'seed'))

        loop.run_until_complete(wp.run())
    except KeyboardInterrupt:
        sys.stderr.flush()
    except:
        logger.exception('error with loop')
    finally:
        loop.close()


if __name__ == '__main__':
    main(mods='/top/all')
