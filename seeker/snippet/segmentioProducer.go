//date: 2022-09-30T17:15:47Z
//url: https://api.github.com/gists/4b64b757660778b3380237c3475324b3
//owner: https://api.github.com/users/AleksandarMuradyan

package main

import (
	"time"
	"github.com/segmentio/kafka-go"
	"fmt"
	"context"
)


func main() {
	topic := "foo"
	partition := 0
	conn, err := kafka.DialLeader(context.Background(), "tcp", "localhost:19091", topic, partition)
	if err != nil {
		fmt.Println("failed to dial leader:", err)
	}

	conn.SetWriteDeadline(time.Now().Add(10*time.Second))
	_, err = conn.WriteMessages(
		kafka.Message{Value: []byte("one!")},
		kafka.Message{Value: []byte("two!")},
		kafka.Message{Value: []byte("three!")},
	)
	if err != nil {
		fmt.Println("failed to write messages:", err)
	}
	if err := conn.Close(); err != nil {
		fmt.Println("failed to close writer:", err)
	}
	fmt.Println("Conn Close")
	fmt.Println(err)
}
