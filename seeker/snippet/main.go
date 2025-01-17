//date: 2025-01-17T16:57:27Z
//url: https://api.github.com/gists/de0e39416ee2ac5cb48ebc333fef1546
//owner: https://api.github.com/users/glutamatt

package main

import (
	"fmt"
	"os"
	"sync/atomic"
	"syscall"
	"time"
)

func main() {
	fileCounter := new(uint64)
	blob := make([]byte, 4096)
	dir := "/tmp/bench_dir"
	fmt.Printf("preparing dir %v ... ", dir)
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		panic(err)
	}
	if err := os.RemoveAll(dir); err != nil {
		panic(err)
	}
	println("ok")
	for i := 0; i < 4; i++ {
		go func() {
			for {
				created, err := os.OpenFile(
					fmt.Sprintf("%s/dumb.%d.txt", dir, atomic.AddUint64(fileCounter, 1)),
					os.O_RDWR|os.O_CREATE|os.O_TRUNC|syscall.O_DIRECT, 0666)
				if err != nil {
					panic(err)
				}
				_, err = created.Write(blob)
				if err != nil {
					panic(err)
				}
				created.Close()
				time.Sleep(time.Second / 4)
			}
		}()
	}

	var prevId uint64
	for {
		time.Sleep(time.Second)
		currentId := atomic.LoadUint64(fileCounter)
		fmt.Printf("files created in %s: %d per sec\n", dir, currentId-prevId)
		prevId = currentId
	}
}
