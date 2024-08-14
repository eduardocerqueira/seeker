//date: 2024-08-14T18:37:39Z
//url: https://api.github.com/gists/6235d69c406bd2e3b16f36974a283fe5
//owner: https://api.github.com/users/alaypatel07

package main

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"sync"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"google.golang.org/grpc"
)

func main() {
	// Define flags for endpoint, CA certificate file, client certificate, key, key prefix, concurrency, duration, and interval
	endpoint := flag.String("etcd-servers", "localhost:2379", "The etcd server endpoint")
	caCertFile := flag.String("etcd-cafile", "", "Path to the CA certificate file")
	certFile := flag.String("etcd-certfile", "", "Path to the client certificate file")
	keyFile := flag.String("etcd-keyfile", "", "Path to the client key file")
	keyPrefix := flag.String("key-prefix", "/foo/", "The prefix for the keys to retrieve")
	concurrency := flag.Int("conc", 5, "Number of concurrent requests")
	duration := flag.Duration("duration", 10*time.Second, "Duration of the test")
	interval := flag.Duration("interval", 1*time.Second, "Frequency at which requests are run")

	flag.Parse()

	// Load the CA certificate
	var tlsConfig *tls.Config
	if *caCertFile != "" {
		certPool := x509.NewCertPool()
		caCert, err := ioutil.ReadFile(*caCertFile)
		if err != nil {
			log.Fatal("Failed to read CA certificate:", err)
		}
		if !certPool.AppendCertsFromPEM(caCert) {
			log.Fatal("Failed to append CA certificate")
		}

		// Load the client certificate and key if provided
		var cert tls.Certificate
		if *certFile != "" && *keyFile != "" {
			cert, err = tls.LoadX509KeyPair(*certFile, *keyFile)
			if err != nil {
				log.Fatal("Failed to load client certificate and key:", err)
			}
		}

		// Configure TLS with the CA cert and client cert if available
		tlsConfig = &tls.Config{
			RootCAs:      certPool,
			Certificates: []tls.Certificate{cert},
		}
	}

	// Connect to the etcd server
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   []string{*endpoint}, // Use the endpoint from the flag
		DialTimeout: 5 * time.Second,
		DialOptions: []grpc.DialOption{
			grpc.WithBlock(),
		},
		TLS: tlsConfig, // Use the TLS configuration with the CA cert and client cert
	})
	if err != nil {
		log.Fatal("Failed to connect to etcd:", err)
	}
	defer cli.Close()

	// Start the concurrent requests based on the flags
	var wg sync.WaitGroup
	endTime := time.Now().Add(*duration)

	for time.Now().Before(endTime) {
		wg.Add(*concurrency)

		for i := 0; i < *concurrency; i++ {
			go func(i int) {
				defer wg.Done()

				// Measure start time
				startTime := time.Now()

				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()

				key := fmt.Sprintf("%s%d", *keyPrefix, i)
				_, err := cli.Put(ctx, key, "bar")
				if err != nil {
					log.Printf("Failed to put key into etcd: %v", err)
					return
				}

				_, err = cli.Get(ctx, key, clientv3.WithPrefix())
				if err != nil {
					log.Printf("Failed to get keys from etcd: %v", err)
					return
				}

				// Measure end time and calculate latency
				endTime := time.Now()
				latency := endTime.Sub(startTime)

				// Print results including start time, end time, and latency
				fmt.Printf("Goroutine %d - Start Time: %s, End Time: %s, Latency: %s\n", i, startTime.Format(time.RFC3339Nano), endTime.Format(time.RFC3339Nano), latency)

				//for _, ev := range resp.Kvs {
				//	fmt.Printf("Goroutine %d: %s : %s\n", i, ev.Key, ev.Value)
				//}
			}(i)
		}

		time.Sleep(*interval)
	}

	wg.Wait()
}
