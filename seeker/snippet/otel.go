//date: 2022-05-02T17:18:14Z
//url: https://api.github.com/gists/db6b1a65d9ff220cbb276a5220236ebe
//owner: https://api.github.com/users/Maixy

package tools

import (
	"context"
	"log"
	"config"

	"github.com/gin-gonic/gin"
	"github.com/spf13/viper"
	"go.opentelemetry.io/contrib/instrumentation/github.com/gin-gonic/gin/otelgin"
	"go.opentelemetry.io/otel"
	stdout "go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.10.0"
)

var requiredOtelConfigurationKeys = [...]string{
	"otel.enabled",
	"otel.servicename",
}

func OtelConfigured() bool {
	logger.Infow("Checking for Otel configuration...")
	if config.GetBool("otel.enabled") {
		for _, value := range requiredOtelConfigurationKeys {
			if !viper.IsSet(value) {
				return false
			}
		}
		logger.Infow("Otel Enabled and all expected OpenTelemetry configurations found!")
	}
	return true
}

func InitOtelTracer() *sdktrace.TracerProvider {
	exporter, err := stdout.New(stdout.WithPrettyPrint())
	if err != nil {
		logger.Fatalw("Error encountered enabling OTel tracer %e", err)
	}
	logger.Debug(exporter)

	resources := resource.NewWithAttributes(
		semconv.SchemaURL,
		semconv.ServiceNameKey.String(config.GetString("otel.servicename")),
		semconv.ServiceVersionKey.String("0.1.0"),
		semconv.ServiceInstanceIDKey.String("SAMPLE_INSTANCE_010203"),
	)

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithSampler(sdktrace.AlwaysSample()),
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(resources),
	)

	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{}, propagation.Baggage{}))

	defer func() {
		if err := tp.Shutdown(context.Background()); err != nil {
			log.Printf("Error shutting down tracer provider: %v", err)
		}
	}()

	return tp
}

func AddOtelMiddlware(router *gin.Engine) {
	router.Use(otelgin.Middleware(config.GetString("otel.servicename")))
}
