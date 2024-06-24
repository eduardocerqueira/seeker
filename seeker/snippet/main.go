//date: 2024-06-24T16:50:59Z
//url: https://api.github.com/gists/191a8d665fb485745186278ff7f3599f
//owner: https://api.github.com/users/danmrichards

package main

import (
	"flag"
	"log"
	"log/slog"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	slidingwindow "github.com/crowdstrike/go-metrics-sliding-window"
	"github.com/rcrowley/go-metrics"
)

var (
	url       string
	sleepRate float64
	sleepTime time.Duration
)

func main() {
	flag.StringVar(&url, "url", "http://www.google.com", "URL to fetch")
	flag.Float64Var(&sleepRate, "sleep-rate", 10, "Percentage of fetch requests to sleep")
	flag.DurationVar(&sleepTime, "sleep-time", 1*time.Second, "Duration to sleep")
	flag.Parse()

	// Create a sample in which to contain the fetch time statistics.
	// Use a sliding window sample which automatically removes samples older
	// then 10 seconds.
	s := slidingwindow.NewSample(1024, time.Second*10)

	// Alternatively, use a uniform sample which keeps all samples forever.
	// s := metrics.NewUniformSample(1028)

	// Create a histogram to hold the sample, and a timer to update the histogram.
	h := metrics.GetOrRegisterHistogram("histogram.latency", metrics.DefaultRegistry, s)
	timer := metrics.NewCustomTimer(h, metrics.NewMeter())

	go fetcher(timer, url, sleepRate, sleepTime)

	go metrics.Log(metrics.DefaultRegistry, 5*time.Second, log.New(os.Stderr, "metrics: ", log.Lmicroseconds))

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig
}

func fetcher(t metrics.Timer, url string, sleepRate float64, sleepTime time.Duration) {
	for {
		now := time.Now()
		if rand.Float64()*100 >= 100-sleepRate {
			time.Sleep(sleepTime)
		}

		if _, err := http.Get(url); err != nil {
			slog.Error("Failed to fetch URL", "url", url, "error", err)
		}

		t.UpdateSince(now)
	}
}
