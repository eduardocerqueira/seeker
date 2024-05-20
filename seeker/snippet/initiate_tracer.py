#date: 2024-05-20T17:11:31Z
#url: https://api.github.com/gists/e04f6b093085f7d1c5ca38aa19ebb27e
#owner: https://api.github.com/users/sylvainkalache

# Set up the TracerProvider
trace.set_tracer_provider(TracerProvider())

# Initialize the ConsoleSpanExporter
console_exporter = ConsoleSpanExporter()

# Set up SimpleSpanProcessor to use ConsoleSpanExporter
span_processor = SimpleSpanProcessor(console_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Get a tracer
tracer = trace.get_tracer(__name__)