#date: 2024-11-13T16:59:29Z
#url: https://api.github.com/gists/35f6a0298fa2c4d6241b678bc2d6b7b5
#owner: https://api.github.com/users/draincoder

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from faststream import FastStream
from faststream.redis import RedisBroker, RedisMessage
from faststream.redis.opentelemetry import RedisTelemetryMiddleware


def get_tracer_provider() -> TracerProvider:
    resource = Resource.create(attributes={"service.name": "faststream"})
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://127.0.0.1:4317")))
    return tracer_provider


broker = RedisBroker(middlewares=(RedisTelemetryMiddleware(tracer_provider=get_tracer_provider()),))
app = FastStream(broker)


@broker.subscriber(channel="export")
async def market_handler(msg: RedisMessage) -> str:
    return f"exported {json.loads(msg.body)['market']}"


if __name__ == "__main__":
    asyncio.run(app.run())
