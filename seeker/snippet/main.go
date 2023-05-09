//date: 2023-05-09T16:56:14Z
//url: https://api.github.com/gists/9819154ae18fc294d151dc6011b6d567
//owner: https://api.github.com/users/henryosei

package main

import (
	"fmt"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	// Block until a signal is received
	sig := <-sigChan
	fmt.Println("Received signal:", sig)
}
