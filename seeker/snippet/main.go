//date: 2023-02-09T17:11:28Z
//url: https://api.github.com/gists/ea58ca27dc9f7f5baa804f556d9cb2d2
//owner: https://api.github.com/users/joaopgrassi

package main

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"time"

	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetrichttp"
	"go.opentelemetry.io/otel/metric/global"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
)

// Mapper function to instrument > temporality
// See: https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/metrics/sdk_exporters/otlp.md#additional-configuration
func DeltaTemporality(ik sdkmetric.InstrumentKind) metricdata.Temporality {
	switch ik {
	case sdkmetric.InstrumentKindCounter, sdkmetric.InstrumentKindObservableCounter, sdkmetric.InstrumentKindHistogram:
		return metricdata.DeltaTemporality
	case sdkmetric.InstrumentKindUpDownCounter, sdkmetric.InstrumentKindObservableUpDownCounter:
		return metricdata.CumulativeTemporality
	default:
		return metricdata.CumulativeTemporality
	}
}

func main() {
	headers := make(map[string]string)
	headers["Authorization"] = "**********"

	eopts := []otlpmetrichttp.Option{
		otlpmetrichttp.WithEndpoint("dynatrace url"),
		otlpmetrichttp.WithURLPath("api/v2/otlp/v1/metrics"),
		otlpmetrichttp.WithHeaders(headers),
		// Configure the exporter with the correct temporality
		otlpmetrichttp.WithTemporalitySelector(DeltaTemporality),
	}

	exporter, err := otlpmetrichttp.New(context.Background(), eopts...)
	if err != nil {
		panic(err)
	}

	meterProvider := sdkmetric.NewMeterProvider(sdkmetric.WithReader(
		sdkmetric.NewPeriodicReader(exporter, sdkmetric.WithInterval(2*time.Second)),
	))

	global.SetMeterProvider(meterProvider)

	meter := global.Meter("my.app")

	counter, err := meter.Int64Counter("my.counter")

	for {
		counter.Add(context.Background(), int64(rand.Intn(20)))
		time.Sleep(time.Second * 5)
	}
}
	}
}
