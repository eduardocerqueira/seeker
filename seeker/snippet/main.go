//date: 2021-10-20T17:00:07Z
//url: https://api.github.com/gists/66a9a1541e542699e0fc6bdf9a91d67e
//owner: https://api.github.com/users/christianfoleide

package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	
	mux := http.NewServeMux()
	
	mux.HandleFunc("/", func(rw http.ResponseWriter, r *http.Request) {
		fmt.Fprint(rw, "Hello, world!")
	})
	
	srv := &http.Server{
		Addr: "host:port",
		Handler: mux,
		ReadTimeout: time.Second * 5,
		WriteTimeout: time.Second * 5,
		IdleTimeout: time.Second * 120,
	}
	
	go func() {
		if err := srv.ListenAndServe(); err != nil {
			if err != http.ErrServerClosed {
				panic(err)
			}
		}
	}()
	
	stopChan := make(chan os.Signal, 1)
	signal.Notify(stopChan, syscall.SIGINT, syscall.SIGTERM)
	
	fmt.Printf("received signal: %+v", <-stopChan)
	
	ctx, cancel := context.WithTimeout(context.Background(), time.Second * 30)
	defer cancel()
	
	if err := srv.Shutdown(ctx); err != nil {
		panic(err)	
	}
}