//date: 2025-05-09T16:51:33Z
//url: https://api.github.com/gists/d75eca0acc8dbbbde20db0901ce4237f
//owner: https://api.github.com/users/sithumonline

package main

import (
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"golang.org/x/net/http2"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "http-two/proto" 
)

const (
	http11Port = ":8080"
	http2Port  = ":8081"
	grpcPort   = ":9090"
)

// --- HTTP/1.1 Server ---
func http11Handler(w http.ResponseWriter, r *http.Request) {
	log.Printf("Hello from HTTP/1.1 at %s! Path: %s\n", r.Host, r.URL.Path)
	//time.Sleep(500 * time.Millisecond) // Simulate some processing time
	_ = simulateWork(45)
	log.Printf("HTTP/1.1 response complete for %s\n", r.URL.Path)
}

func serveHTTP11() {
	http.HandleFunc("/1.1/", http11Handler)
	server := &http.Server{Addr: http11Port}
	log.Printf("HTTP/1.1 server listening on %s", http11Port)
	go func() {
		if err := server.ListenAndServe(); err != http.ErrServerClosed {
			log.Fatalf("HTTP/1.1 server ListenAndServe error: %v", err)
		}
	}()
}

// --- HTTP/2 Server (without gRPC) ---
func http2Handler(w http.ResponseWriter, r *http.Request) {
	log.Printf("Hello from HTTP/2 at %s! Path: %s\n", r.Host, r.URL.Path)
	//time.Sleep(300 * time.Millisecond) // Simulate slightly faster processing
	_ = simulateWork(45)
	log.Printf("HTTP/2 response complete for %s\n", r.URL.Path)
}

func serveHTTP2() {
	server := &http.Server{Addr: http2Port}
	if err := http2.ConfigureServer(server, &http2.Server{}); err != nil {
		log.Fatalf("Failed to configure HTTP/2 server: %v", err)
	}
	http.HandleFunc("/2/", http2Handler)
	log.Printf("HTTP/2 server listening on %s", http2Port)
	go func() {
		if err := server.ListenAndServeTLS("server.crt", "server.key"); err != http.ErrServerClosed {
			log.Fatalf("HTTP/2 server ListenAndServeTLS error: %v", err)
		}
	}()
}

// --- gRPC Server ---
type GreeterServer struct {
	pb.UnimplementedGreeterServer
}

func (s *GreeterServer) SayHello(req *pb.HelloRequest, resp pb.Greeter_SayHelloServer) error {
	log.Printf("Received gRPC SayHello request: %v", req.GetName())
	for i := 0; i < 3; i++ {
		// time.Sleep(200 * time.Millisecond) // Simulate streaming
		n := simulateWork(45)
		// if err := resp.Send(&pb.HelloReply{Message: fmt.Sprintf("Hello %s, stream %d!", req.GetName(), i+1)}); err != nil {
		// 	return err
		// }
		if err := resp.Send(&pb.HelloReply{Message: fmt.Sprintf("Hello %s, stream %d! %d", req.GetName(), i+1, n)}); err != nil {
			return err
		}
	}
	return nil
}

func serveGRPC() {
	lis, err := net.Listen("tcp", grpcPort)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	s := grpc.NewServer()
	pb.RegisterGreeterServer(s, &GreeterServer{})
	log.Printf("gRPC server listening on %s", grpcPort)
	go func() {
		if err := s.Serve(lis); err != nil {
			log.Fatalf("failed to serve gRPC: %v", err)
		}
	}()
}

// simulateWork performs a time-consuming calculation (e.g., Fibonacci).
func simulateWork(n int) int {
	if n <= 1 {
		return n
	}
	return simulateWork(n-1) + simulateWork(n-2)
}

// --- Clients ---
func http11Client(urlStr string) {
	fmt.Printf("\n--- HTTP/1.1 Client requesting %s ---\n", urlStr)
	start := time.Now()
	resp, err := http.Get(urlStr)
	if err != nil {
		log.Printf("HTTP/1.1 GET error: %v", err)
		return
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("HTTP/1.1 ReadAll error: %v", err)
		return
	}
	fmt.Printf("HTTP/1.1 Response (%d): %s (took %v)\n", resp.StatusCode, string(body), time.Since(start))
}

func http2Client(urlStr string) {
	fmt.Printf("\n--- HTTP/2 Client requesting %s ---\n", urlStr)
	client := &http.Client{
		Transport: &http2.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true}, // For self-signed certs
		},
	}
	start := time.Now()
	resp, err := client.Get(urlStr)
	if err != nil {
		log.Printf("HTTP/2 GET error: %v", err)
		return
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("HTTP/2 ReadAll error: %v", err)
		return
	}
	fmt.Printf("HTTP/2 Response (%d): %s (took %v)\n", resp.StatusCode, string(body), time.Since(start))
}

func grpcClient(target string, name string) {
	fmt.Printf("\n--- gRPC Client calling SayHello on %s with name '%s' ---\n", target, name)
	conn, err := grpc.Dial(target, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()
	c := pb.NewGreeterClient(conn)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second*15)
	defer cancel()

	stream, err := c.SayHello(ctx, &pb.HelloRequest{Name: name})
	if err != nil {
		log.Fatalf("could not SayHello: %v", err)
	}
	for {
		reply, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatalf("failed to receive: %v", err)
		}
		log.Printf("gRPC Response: %s\n", reply.GetMessage())
	}
}

func main() {
	// --- Generate self-signed certificates for HTTP/2 ---
	// You'll need to generate server.crt and server.key.
	// A simple way is using openssl:
	// openssl genrsa -out server.key 2048
	// openssl req -new -x509 -key server.key -out server.crt -days 365

	// Check if certificate files exist
	if _, err := os.Stat("server.crt"); os.IsNotExist(err) {
		log.Println("Warning: server.crt not found. HTTP/2 server will likely fail.")
		log.Println("Generate self-signed certificates using openssl:")
		log.Println("  openssl genrsa -out server.key 2048")
		log.Println("  openssl req -new -x509 -key server.key -out server.crt -days 365")
	}
	if _, err := os.Stat("server.key"); os.IsNotExist(err) {
		log.Println("Warning: server.key not found.")
	}

	// --- Start Servers ---
	serveHTTP11()
	serveHTTP2()
	serveGRPC()

	// --- Wait for servers to start ---
	time.Sleep(time.Second)

	// --- Run Clients in parallel ---
	baseURL11 := &url.URL{Scheme: "http", Host: "localhost" + http11Port, Path: "/1.1/"}
	baseURL2 := &url.URL{Scheme: "https", Host: "localhost" + http2Port, Path: "/2/"}
	grpcTarget := "localhost" + grpcPort

	var wg sync.WaitGroup
	wg.Add(5) // Total number of requests

	// HTTP/1.1 requests
	go func() {
		defer wg.Done()
		http11Client(baseURL11.String() + "request1")
	}()
	go func() {
		defer wg.Done()
		http11Client(baseURL11.String() + "request2")
	}()

	// HTTP/2 requests
	go func() {
		defer wg.Done()
		http2Client(baseURL2.String() + "requestA")
	}()
	go func() {
		defer wg.Done()
		http2Client(baseURL2.String() + "requestB")
	}()

	// gRPC request
	go func() {
		defer wg.Done()
		grpcClient(grpcTarget, "Go User")
	}()

	// Wait for all requests to complete
	wg.Wait()

	// --- Keep the servers running until interrupted ---
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan
	log.Println("Shutting down servers...")
}
