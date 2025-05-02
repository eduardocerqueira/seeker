//date: 2025-05-02T16:34:19Z
//url: https://api.github.com/gists/9b787614dce96202d9278fba1e98a1b2
//owner: https://api.github.com/users/Kiand1

package main

import (
	"fmt"
	"net"
)

// 实现一个端口扫描器
func main() {
	for i := 0; i < 65535; i++ {
		addr := fmt.Sprintf("127.0.0.1:%d", i)
		dial, err := net.Dial("tcp", addr)
		if err != nil {
			continue
		}
		dial.Close()
		fmt.Println("开启", i)
	}
}
