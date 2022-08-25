//date: 2022-08-25T16:54:21Z
//url: https://api.github.com/gists/7c039af438ee25eadf02bce274bd8d07
//owner: https://api.github.com/users/isparo

package main

import (
	"context"
	"fmt"
	"log"
	"net"

	"github.com/josue/grpc_golang_tutorial/unary_example/server/protofiles/greetpb"
	"google.golang.org/grpc"
)

// this struct could be used to dependency injection
type server struct{}

// handler
func (s server) Greet(ctx context.Context, req *greetpb.GreetRequest) (*greetpb.GreetResponse, error) {

	log.Println("User name: ", req.UserName)
	log.Println("Country code: ", req.CountryCode)

	var greeting string

	switch req.CountryCode {
	case "us":
		greeting = "hello " + req.UserName
	case "mx":
		greeting = "Hola " + req.UserName
	default:
		greeting = "Hola/Hello " + req.UserName
	}

	return &greetpb.GreetResponse{
		Result: greeting,
	}, nil
}

func main() {
	listener, err := net.Listen("tcp", "localhost:8080")

	if err != nil {
		panic(err)
	}

	fmt.Println("starting server")

	s := grpc.NewServer()
	greetpb.RegisterGreetServiceServer(s, server{})

	if err := s.Serve(listener); err != nil {
		panic(err)
	}
}
