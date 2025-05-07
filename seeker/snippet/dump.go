//date: 2025-05-07T17:10:29Z
//url: https://api.github.com/gists/b55c33677646a44290045897b53562ef
//owner: https://api.github.com/users/paulja

package main

import (
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"
)

func main() {
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGQUIT)
	cancel := make(chan bool)
	go func() {
		fmt.Print("running")
		for {
			<-time.After(time.Second)
			fmt.Print(".")
		}
	}()
	go func() {
		<-sig
		fmt.Printf("\nstopping...\n")
		buf := make([]byte, 1<<20)
		n := runtime.Stack(buf, true)
		fmt.Printf("\n*** goroutine dump\n%s\n*** end\n", buf[:n])
		cancel <- true
	}()
	<-cancel
	fmt.Println("done")
}
