//date: 2022-08-26T17:16:12Z
//url: https://api.github.com/gists/d70eebd8b0e6bb1a4578ac7c06a842cf
//owner: https://api.github.com/users/ryepup

package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	variants := map[string]string{
		"otel":                    "otel-collector:4317",
		"otel with tail sampling": "otel-collector:4318",
	}

	for name, endpoint := range variants {
		tp, stop := createProvider(ctx, endpoint)
		defer stop()
		_, span := tp.Tracer(name).Start(ctx, name,
			trace.WithAttributes(attribute.String("custom", name)),
		)
		span.End()
	}
}

func createProvider(ctx context.Context, endpoint string) (*sdktrace.TracerProvider, func()) {
	exporter, err := otlptracegrpc.New(ctx,
		otlptracegrpc.WithInsecure(),
		otlptracegrpc.WithEndpoint(endpoint),
	)
	if err != nil {
		panic(err)
	}
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithSampler(sdktrace.AlwaysSample()),
		sdktrace.WithBatcher(exporter),
	)
	stop := func() {
		_ = tp.ForceFlush(context.Background())
		_ = tp.Shutdown(context.Background())
	}
	return tp, stop
}
