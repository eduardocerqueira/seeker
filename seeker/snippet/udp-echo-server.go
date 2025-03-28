//date: 2025-03-28T16:45:50Z
//url: https://api.github.com/gists/50e8f53a1d778e66186d9ac9b2af55da
//owner: https://api.github.com/users/winlinvip

package main

import (
	"flag"
	"fmt"
	"net"
)

func udpEchoServer(host string, port int) {
	addr := fmt.Sprintf("%s:%d", host, port)
	udpAddr, err := net.ResolveUDPAddr("udp", addr)
	if err != nil {
		fmt.Println("Error resolving UDP address:", err)
		return
	}

	conn, err := net.ListenUDP("udp", udpAddr)
	if err != nil {
		fmt.Println("Error listening:", err)
		return
	}
	defer conn.Close()

	fmt.Printf("UDP Echo Server is listening on %s...\n", addr)

	buf := make([]byte, 1024)

	for {
		n, remoteAddr, err := conn.ReadFromUDP(buf)
		if err != nil {
			fmt.Println("Error receiving data:", err)
			continue
		}

		fmt.Printf("Received from %s: %s\n", remoteAddr, string(buf[:n]))

		_, err = conn.WriteToUDP(buf[:n], remoteAddr)
		if err != nil {
			fmt.Println("Error sending data:", err)
		}
	}
}

func main() {
	host := flag.String("host", "0.0.0.0", "Host to listen on")
	port := flag.Int("port", 20099, "Port to listen on")
	flag.Parse()

	udpEchoServer(*host, *port)
}
