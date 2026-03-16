#date: 2026-03-16T17:47:33Z
#url: https://api.github.com/gists/48ec1909e753b9e8c9c0c1b8547174a6
#owner: https://api.github.com/users/muellerzr-lambda

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["aiohttp"]
# ///
"""Server to serve chat.html and proxy /v1/ requests to multiple vLLM backends."""

import json
import aiohttp
from aiohttp import web

BACKENDS = [
    "http://132.145.137.59:8001",
    "http://132.145.137.59:8002",
]
STATIC_FILE = "/home/ubuntu/gtc_demo/mle-conference-demos/lambda-latest-models/chat.html"

# Cache: model_name -> backend_url
model_backend_map = {}


async def refresh_model_map():
    """Query all backends and build model -> backend mapping."""
    new_map = {}
    async with aiohttp.ClientSession() as session:
        for backend in BACKENDS:
            try:
                async with session.get(f"{backend}/v1/models", timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for m in data.get("data", []):
                            new_map[m["id"]] = backend
            except Exception:
                pass
    model_backend_map.update(new_map)
    return new_map


def get_backend_for_model(model_name):
    """Look up which backend serves a given model."""
    return model_backend_map.get(model_name, BACKENDS[0])


async def handle_root(request):
    return web.FileResponse(STATIC_FILE)


async def proxy_vllm_get(request: web.Request):
    path = request.match_info["path"]

    # For /v1/models, aggregate from all backends
    if path == "models":
        await refresh_model_map()
        all_models = []
        seen = set()
        async with aiohttp.ClientSession() as session:
            for backend in BACKENDS:
                try:
                    async with session.get(
                        f"{backend}/v1/models",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            for m in data.get("data", []):
                                if m["id"] not in seen:
                                    seen.add(m["id"])
                                    m["backend"] = backend
                                    all_models.append(m)
                except Exception:
                    pass
        return web.json_response({"object": "list", "data": all_models})

    # Other GET requests: try first backend
    url = f"{BACKENDS[0]}/v1/{path}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.read()
            return web.Response(
                body=data,
                status=resp.status,
                content_type=resp.headers.get("Content-Type", "application/json"),
            )


async def proxy_vllm_post(request: web.Request):
    path = request.match_info["path"]
    body_bytes = await request.read()

    # Determine which backend to use based on model in request body
    backend = BACKENDS[0]
    try:
        body_json = json.loads(body_bytes)
        model = body_json.get("model", "")
        backend = get_backend_for_model(model)
    except (json.JSONDecodeError, KeyError):
        pass

    url = f"{backend}/v1/{path}"
    headers = {"Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=body_bytes, headers=headers) as resp:
            ct = resp.headers.get("Content-Type", "")
            if "text/event-stream" in ct:
                response = web.StreamResponse(
                    status=resp.status,
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )
                await response.prepare(request)
                async for chunk in resp.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                return response
            else:
                data = await resp.read()
                return web.Response(
                    body=data,
                    status=resp.status,
                    content_type=ct or "application/json",
                )


app = web.Application()
app.router.add_get("/", handle_root)
app.router.add_get("/v1/{path:.*}", proxy_vllm_get)
app.router.add_post("/v1/{path:.*}", proxy_vllm_post)

if __name__ == "__main__":
    print("Chat UI running at http://localhost:9001")
    print(f"Proxying to backends: {BACKENDS}")
    web.run_app(app, host="0.0.0.0", port=9001)