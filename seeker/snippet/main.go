//date: 2022-03-04T17:13:42Z
//url: https://api.github.com/gists/47594bd1367bd0919b0faac13d19200e
//owner: https://api.github.com/users/NomNomCameron

package main

import (
	"fmt"
	"log"
	"time"

	"github.com/joho/godotenv"
)

func main() {
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env")
	}

	dataChan := make(chan []map[string]interface{})

	tvSocket := TVSocket{
		ConnStatus:    make(chan string),
		LastHeartBeat: time.Now(),
		DataChan:      dataChan,
	}

	pc := PositionCollection{
		TVSocket: &tvSocket,
	}
	go func() {
		for {
			select {
			case body := <-tvSocket.DataChan:
				err := pc.HandleData(body)
				if err != nil {
					fmt.Println("UNEXPECTED ERROR:", err)
				} else {
                                        pc.HandleData(body)
                                }
			}
		}
	}()

	go tvSocket.Connect("access-token", "wss://url-to-ws.com")
	connStatus := <-tvSocket.ConnStatus
        if connStatus != "Ready" {
                fmt.Println("conn issue:", connStatus)
        }
        pc.AddOrder(...) // Adds 1 to pc.WaitGroup - writes to tvsocket and should trigger multiple messages to be sent over pc.TVSocket.DataChan 
        pc.WaitGroup.Wait()
        pc.AddOrder(...) // Adds 1 to pc.WaitGroup - writes to tvsocket and should trigger multiple messages to be sent over pc.TVSocket.DataChan
        pc.WaitGroup.Wait()
}