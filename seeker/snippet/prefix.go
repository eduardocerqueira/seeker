//date: 2021-08-31T02:56:04Z
//url: https://api.github.com/gists/9eed94230db53cc942ad3e2ca3db6909
//owner: https://api.github.com/users/chadleeshaw

package main

import (
	"fmt"
	"net"
)

func main() {
	mask := "255.255.248.0"
	stringMask := net.IPMask(net.ParseIP(mask).To4())
	length, _ := stringMask.Size()
	fmt.Printf("Subnetmask is a /%d\n", length)
}