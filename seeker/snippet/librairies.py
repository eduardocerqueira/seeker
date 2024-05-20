#date: 2024-05-20T17:11:31Z
#url: https://api.github.com/gists/e04f6b093085f7d1c5ca38aa19ebb27e
#owner: https://api.github.com/users/sylvainkalache

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor