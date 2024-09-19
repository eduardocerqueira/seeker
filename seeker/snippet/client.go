//date: 2024-09-19T16:41:24Z
//url: https://api.github.com/gists/5598eded0409e47acaf5cdcf7855f591
//owner: https://api.github.com/users/loderunner

package main

import (
	"fmt"
	"math/rand"
	"net/http"
	"time"

	"github.com/gorilla/websocket"
)

type DataPoint struct {
	Timestamp   time.Time `json:"timestamp"`
	Temperature float32   `json:"temperature"`
}

func main() {
	conn, _, err := websocket.DefaultDialer.Dial("ws://localhost:8888/data", http.Header{})
	if err != nil {
		panic(err.Error())
	}
	ticker := time.Tick(2 * time.Second)
	for {
		select {
		case <-ticker:
			dataPoint := DataPoint{
				Timestamp:   time.Now(),
				Temperature: float32(rand.NormFloat64()*5 + 20),
			}
			fmt.Printf("[%s] %.2fºC\n",
				dataPoint.Timestamp.Format(time.RFC3339),
				dataPoint.Temperature,
			)
			err = conn.WriteJSON(dataPoint)
			if err != nil {
				panic(err.Error())
			}
		}
	}
}
