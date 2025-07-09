#date: 2025-07-09T17:04:10Z
#url: https://api.github.com/gists/44cd94756d5556a1fd4fe511a03b780e
#owner: https://api.github.com/users/dpguthrie

#!/usr/bin/env python3
"""
Minimal reproducible example: Sending OTEL spans to different Braintrust projects in a single batch.

This example demonstrates how to use the 'braintrust.parent' span attribute to route
different spans to different projects, even when they're sent in the same HTTP batch.
"""

import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configuration - Replace with your actual values
BRAINTRUST_API_URL = os.getenv("BRAINTRUST_API_URL", "https://api.braintrust.dev")
BRAINTRUST_API_KEY = os.getenv("BRAINTRUST_API_KEY")  # Set this environment variable

# Project IDs for different apps/services
PROJECT_A_ID = "84d5f003-a4db-42c1-82f4-567531d36fae"  # Replace with actual project ID
PROJECT_B_ID = "683e584a-d84d-494e-8ade-4c6cf25d5a65"  # Replace with actual project ID
DEFAULT_PROJECT_ID = PROJECT_A_ID


def setup_otel_tracer():
    """Set up OpenTelemetry tracer with Braintrust exporter."""

    if not BRAINTRUST_API_KEY:
        raise ValueError("Please set BRAINTRUST_API_KEY environment variable")

    # Configure the OTLP exporter
    exporter = OTLPSpanExporter(
        endpoint=f"{BRAINTRUST_API_URL}/otel/v1/traces",
        headers={
            "Authorization": f"Bearer {BRAINTRUST_API_KEY}",
            # This header provides a fallback project for spans without braintrust.parent attribute
            "x-bt-parent": f"project_id:{DEFAULT_PROJECT_ID}",
        },
    )

    # Set up tracer provider with batch processor
    provider = TracerProvider()
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    return provider, trace.get_tracer("multi-project-example")


def create_spans_for_different_projects(tracer):
    """Create spans that will be routed to different projects."""

    print("Creating spans for different projects...")

    # Span 1: Explicitly route to Project A
    with tracer.start_as_current_span("user_authentication") as span:
        span.set_attribute("braintrust.parent", f"project_id:{PROJECT_A_ID}")
        span.set_attribute("app", "user-service")
        span.set_attribute("operation", "login")
        span.set_attribute("user_id", "user123")
        span.set_attribute("success", True)
        print(f"  - Created span 'user_authentication' for Project A: {PROJECT_A_ID}")

    # Span 2: Explicitly route to Project B
    with tracer.start_as_current_span("payment_processing") as span:
        span.set_attribute("braintrust.parent", f"project_id:{PROJECT_B_ID}")
        span.set_attribute("app", "payment-service")
        span.set_attribute("operation", "charge")
        span.set_attribute("amount", 99.99)
        span.set_attribute("currency", "USD")
        print(f"  - Created span 'payment_processing' for Project B: {PROJECT_B_ID}")

    # Span 3: Another span for Project A
    with tracer.start_as_current_span("profile_update") as span:
        span.set_attribute("braintrust.parent", f"project_id:{PROJECT_A_ID}")
        span.set_attribute("app", "user-service")
        span.set_attribute("operation", "update_profile")
        span.set_attribute("user_id", "user123")
        span.set_attribute("fields_updated", ["email", "phone"])
        print(f"  - Created span 'profile_update' for Project A: {PROJECT_A_ID}")

    # Span 4: Use default project (no braintrust.parent attribute)
    with tracer.start_as_current_span("system_health_check") as span:
        span.set_attribute("app", "monitoring-service")
        span.set_attribute("operation", "health_check")
        span.set_attribute("status", "healthy")
        print(f"  - Created span 'system_health_check' for default project: {DEFAULT_PROJECT_ID}")

    # Span 5: Route to Project B using project name instead of ID
    # Note: This requires the project name to be resolvable by Braintrust
    with tracer.start_as_current_span("order_fulfillment") as span:
        span.set_attribute("braintrust.parent", f"project_id:{PROJECT_B_ID}")
        span.set_attribute("app", "order-service")
        span.set_attribute("operation", "fulfill_order")
        span.set_attribute("order_id", "order456")
        span.set_attribute("items_count", 3)
        print(f"  - Created span 'order_fulfillment' for Project B: {PROJECT_B_ID}")


def main():
    """Main function demonstrating multi-project OTEL spans."""

    print("=== Braintrust Multi-Project OTEL Example ===\n")

    # Validate configuration
    if not all([PROJECT_A_ID, PROJECT_B_ID, DEFAULT_PROJECT_ID]):
        print("ERROR: Please update PROJECT_A_ID, PROJECT_B_ID, and DEFAULT_PROJECT_ID with your actual project IDs")
        return

    try:
        # Set up OpenTelemetry
        provider, tracer = setup_otel_tracer()
        print("✓ OpenTelemetry tracer configured with Braintrust exporter\n")

        # Create spans for different projects
        create_spans_for_different_projects(tracer)

        # Force flush to send all spans immediately
        print("\nFlushing spans to Braintrust...")
        provider.force_flush()
        print("✓ Spans sent successfully!")

        print(f"\nResult:")
        print(f"- Spans will appear in their respective projects in Braintrust")
        print(f"- Project A ({PROJECT_A_ID}): user_authentication, profile_update")
        print(f"- Project B ({PROJECT_B_ID}): payment_processing, order_fulfillment")
        print(f"- Default Project ({DEFAULT_PROJECT_ID}): system_health_check")
        print(f"\nCheck your Braintrust projects to verify the spans were routed correctly!")

    except Exception as e:
        print(f"ERROR: {e}")
        return


if __name__ == "__main__":
    main()
