//date: 2021-12-03T17:08:36Z
//url: https://api.github.com/gists/b5618263c06ff5574ddec7fbb5f5824e
//owner: https://api.github.com/users/shakahl

package main

import (
	"fmt"
	"log"
	"net"
	"net/http"
	"net/rpc"
	"time"
)

type Args struct {
	A int
	B int
}

type Calculator int

func (calc *Calculator) Add(args Args, reply *int) error {
	*reply = args.A + args.B
	return nil
}

func main() {
	go startHttpRpcServer()

	client, err := rpc.DialHTTP("tcp", "localhost:1234") // <-- connects to an HTTP RPC server at the specified network address listening on the default HTTP RPC path.
	if err != nil {
		log.Fatal("dialing:", err)
	}
	args := Args{5, 3}

	var reply int
	err = client.Call("Calculator.Add", args, &reply)
	if err != nil {
		log.Fatal("arith error:", err)
	}

	fmt.Printf("Calculation %d+%d=%d\n", args.A, args.B, reply)
}

func startHttpRpcServer() {
	func() {
		calculator := new(Calculator)
		rpc.Register(calculator) // <-- Register publishes the receiver's methods in the DefaultServer.
		rpc.HandleHTTP()         // <-- Registers an HTTP handler for RPC messages
		listener, e := net.Listen("tcp", ":1234") // <-- creates an tcp listener on localhost
		if e != nil {
			log.Fatal("listen error:", e)
		}
		fmt.Println("Starting serving rpc")
		time.Sleep(2 * time.Second)
		go http.Serve(listener, nil) // <-- Starts accepting tcp connections to the listener
		fmt.Println("Serving rpc")
	}()
}
