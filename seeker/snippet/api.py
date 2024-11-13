#date: 2024-11-13T16:59:29Z
#url: https://api.github.com/gists/35f6a0298fa2c4d6241b678bc2d6b7b5
#owner: https://api.github.com/users/draincoder

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from faststream.redis import RedisMessage
from faststream.redis.fastapi import RedisRouter
from faststream.redis.opentelemetry import RedisTelemetryMiddleware

router = RedisRouter()


@router.get("/export/{market}")
async def market_handler(market: str) -> str:
    response: RedisMessage = await router.broker.request(
        {"market": market},
        channel="export",
        timeout=40.0,
    )
    return response.body.decode()


def create_app() -> FastAPI:
    resource = Resource.create(attributes={"service.name": "fastapi"})
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://127.0.0.1:4317")))

    app = FastAPI(debug=True)
    app.include_router(router)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)
    router.broker.add_middleware(RedisTelemetryMiddleware(tracer_provider=tracer_provider))

    return app


if __name__ == "__main__":
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
