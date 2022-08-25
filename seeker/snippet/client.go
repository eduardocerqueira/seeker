//date: 2022-08-25T16:54:21Z
//url: https://api.github.com/gists/7c039af438ee25eadf02bce274bd8d07
//owner: https://api.github.com/users/isparo

package main

import (
	"context"
	"log"

	"github.com/josue/grpc_golang_tutorial/unary_example/server/protofiles/greetpb"
	"google.golang.org/grpc"
)

func main() {
	cc, err := grpc.Dial("localhost:8080", grpc.WithInsecure())

	if err != nil {
		panic(err)
	}

	defer cc.Close()

	c := greetpb.NewGreetServiceClient(cc)

	getGreeting("Jack", "us", c)
	getGreeting("Jose", "mx", c)

}

func getGreeting(name, countryCode string, c greetpb.GreetServiceClient) {

	log.Println("creating greeting")

	res, err := c.Greet(context.Background(), &greetpb.GreetRequest{
		CountryCode: countryCode,
		UserName:    name,
	})

	if err != nil {
		log.Println("error: ", err)
		panic(err)
	}

	log.Println(res.Result)
}
