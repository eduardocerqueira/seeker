//date: 2025-04-30T16:50:29Z
//url: https://api.github.com/gists/71131af1a62907ed360452c2507c1cc8
//owner: https://api.github.com/users/coalaura

package main

import (
	"fmt"
	"net"
	"os"
	"strconv"
	"time"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Println("Usage: tcp <ip> <port>")

		os.Exit(1)
	}

	ip := net.ParseIP(os.Args[1])
	if ip == nil || ip.IsUnspecified() {
		fmt.Println("invalid ip address")

		os.Exit(1)
	}

	port, err := strconv.ParseInt(os.Args[2], 10, 64)
	if err != nil || port < 1 || port > 65535 {
		fmt.Println("invalid port")

		os.Exit(1)
	}

	address := net.JoinHostPort(ip.String(), fmt.Sprint(port))

	start := time.Now()
	conn, err := net.DialTimeout("tcp", address, 10*time.Second)
	taken := time.Since(start)

	if err != nil {
		fmt.Printf("failed: 127.0.0.1 -/- %s after %s\n", address, round(taken))
		fmt.Println(err.Error())

		os.Exit(1)
	}

	conn.Close()

	fmt.Printf("success: %s <-> %s after %s\n", conn.LocalAddr().String(), conn.RemoteAddr().String(), round(taken))
}

func round(d time.Duration) time.Duration {
	switch {
	case d > time.Minute:
		return d.Round(time.Second)
	case d > time.Second:
		return d.Round(100 * time.Millisecond)
	case d > time.Millisecond:
		return d.Round(10 * time.Microsecond)
	case d > time.Microsecond:
		return d.Round(10 * time.Nanosecond)
	}

	return d
}
