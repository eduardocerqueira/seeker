//date: 2022-07-21T17:07:13Z
//url: https://api.github.com/gists/366497ad130f0afa8b47126c727c1217
//owner: https://api.github.com/users/dongnguyenltqb

package main

import (
	"flag"
	"fmt"
	"io"
	"net"
)

var port int
var remoteAddress string

func init() {
	fmt.Println("Parsing config..")
	flag.IntVar(&port, "port", 8080, "local port to listen")
	flag.StringVar(&remoteAddress, "remote", "home.dongnguyen.dev:80", "remote address to forward")
	flag.Parse()
}

func main() {
	fmt.Printf("Start listening on port: %d\n", port)
	ln, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		panic(err)
	}
	for {
		conn, err := ln.Accept()
		if err != nil {
			panic(err)
		}

		go handleRequest(conn)
	}
}

func handleRequest(conn net.Conn) {
	fmt.Println("new client connected.")
	proxy, err := net.Dial("tcp", remoteAddress)
	if err != nil {
		panic(err)
	}
	fmt.Println("proxy connected")
	go copyIO(conn, proxy)
	go copyIO(proxy, conn)
}

func copyIO(src, dest net.Conn) {
	defer src.Close()
	defer dest.Close()
	io.Copy(src, dest)
}
