//date: 2022-01-17T17:09:53Z
//url: https://api.github.com/gists/e296c09082b65ba3931aecbc118876ba
//owner: https://api.github.com/users/VirusBreeder

package main

import (
	"fmt"
	"log"
	"net"
	"time"
)

func accept_connection(conn net.Conn) {
	err := conn.SetReadDeadline(time.Now().Add(5 * time.Second))
	if err != nil {
		panic(err)
	}
	bb := make([]byte, 1024)
	_, err = conn.Read(bb[:])
	if err != nil {
		if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
			panic(err)
		} else {
			panic(err)
		}
	}
	err = conn.Close()
	if err != nil {
		log.Println(err)
	}
	log.Printf("IMCOMMING MESSAGE: %s", string(bb))
}

func listen() {
	ln, err := net.Listen("tcp", "127.0.0.1:8081")
	if err != nil {
		panic(err)
	}
	for {
		conn, err := ln.Accept()
		if err != nil {
			panic(err)
		}
		go accept_connection(conn)
	}
}

func dial(message []byte) {
	conn, err := net.Dial("tcp", "127.0.0.1:8081")
	if err != nil {
		panic(err)
	}
	_, err = conn.Write(message)
	if err != nil {
		panic(err)
	}
	err = conn.Close()
	if err != nil {
		panic(err)
	}
}

func sleep(duration float64) {
	time.Sleep(time.Millisecond * time.Duration(duration*1000))
}

func main() {
	go listen()
	sleep(0.1)
	for i := 0; i < 10; i++ {
		dial([]byte(fmt.Sprintf("hello world %d", i)))
		sleep(0.3)
	}
}
