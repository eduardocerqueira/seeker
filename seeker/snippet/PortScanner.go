//date: 2025-02-11T16:49:54Z
//url: https://api.github.com/gists/ab420b11b153844aa3613def8d239b51
//owner: https://api.github.com/users/xandersavvy

package main

import (
	"fmt"
	"net"
	"sync"
	"time"
)

func scanPort(host string, port int, wg *sync.WaitGroup) {
	defer wg.Done()
	address := fmt.Sprintf("%s:%d", host, port)
	conn, err := net.DialTimeout("tcp", address, 1*time.Second)
	if err == nil {
		fmt.Printf("[+] Port %d is open\n", port)
		conn.Close()
	}
}

func main() {
	var host string
	var startPort, endPort int

	fmt.Print("Enter target host (IP or domain): ")
	fmt.Scanln(&host)
	fmt.Print("Enter start port: ")
	fmt.Scanln(&startPort)
	fmt.Print("Enter end port: ")
	fmt.Scanln(&endPort)

	var wg sync.WaitGroup

	for port := startPort; port <= endPort; port++ {
		wg.Add(1)
		go scanPort(host, port, &wg)
	}

	wg.Wait()
}
// Things to learn not from GPT 😂
/****
*printf just prints but Sprintf print and also return the formatted string which can be stored later
*Waitgroup to add go routine so main function does not exit unless every go routine is resolved 
*net.DialTImeout to establish connection
******/
/***
*Hey if you are here and have some development project please feel free to contact me on savyxander7@gmail.com
****/


