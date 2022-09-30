//date: 2022-09-30T17:15:47Z
//url: https://api.github.com/gists/4b64b757660778b3380237c3475324b3
//owner: https://api.github.com/users/AleksandarMuradyan

package main

import (
	"context"
	"fmt"
	"time"
	"github.com/segmentio/kafka-go"
)

func main() {
	topic := "foo"
	partition := 0
	conn, err := kafka.DialLeader(context.Background(), "tcp", "localhost:19091", topic, partition)
	if err != nil {
		fmt.Println("failed to dial leader:", err)
	}
	conn.SetReadDeadline(time.Now().Add(10*time.Second))
	batch := conn.ReadBatch(10e3, 1e6) // fetch 10KB min, 1MB max
	b := make([]byte, 10e3) // 10KB max per message
	for {
		_, err := batch.Read(b)
		if err != nil {
			fmt.Println(err)
			break
		}
		fmt.Println(string(b))
	}
	
	if err := batch.Close(); err != nil {
		fmt.Println("failed to close batch:", err)
	}
	
	if err := conn.Close(); err != nil {
		fmt.Println("failed to close connection:", err)
	}
}