//date: 2022-02-04T16:55:34Z
//url: https://api.github.com/gists/a763281e26d1b4434f47c14a293d10e1
//owner: https://api.github.com/users/JankyGaming

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/docker/go-connections/nat"
	"github.com/nats-io/nats.go"
	tc "github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/wait"
	"gotest.tools/assert"
	"strings"
	"testing"
	"time"
)

var serviceName = "myServer"
var streamName = "something"
var streamSubject = "something.>"

func TestWithDrainInCallback(t *testing.T) {
	ctx := context.Background()
	testNats := func(t *testing.T, numberOfEvents int, numberOfDrains int) {
		// Starts a nats container to connect to
		connStr := startNatsContainer(ctx)

		// Configure nats and JS client
		jsc := setUpJetStream(t, connStr, streamName, streamSubject, serviceName)

		// Publish events as fast as possible in a go routine, this happens concurrently to the multiple
		//  Subscribes and drains
		go func() {
			for i := 0; i < numberOfEvents; i++ {
				marshal, _ := json.Marshal(map[string]interface{}{
					"number": i+1,
				})

				_, err := jsc.Publish("something.anything", marshal)
				if err != nil {
					panic(err)
				}
			}
		}()

		// Map to verify integrity of events
		messageMap := map[int]*struct{}{}
		// Slice to count number of messages received
		var receivedMessages []*nats.Msg
		// Int to ALSO count number of messages I receive
		msgCount := 0

		// This will never complete because nats appears to be dropping messages despite using .Drain()
		for len(receivedMessages) <= numberOfEvents {
			subscription, err := jsc.QueueSubscribe("something.>", serviceName, func(msg *nats.Msg) {
				var dataMap map[string]interface{}
				json.Unmarshal(msg.Data, &dataMap)

				// Add and increment data for event
				numberInMessage := int(dataMap["number"].(float64))
				if _, ok := messageMap[numberInMessage]; ok {
					t.Fatal(numberInMessage, "is in list of events more than once")
				}
				messageMap[numberInMessage] = nil

				receivedMessages = append(receivedMessages, msg)

				msgCount++

				// Every 10% of events, Drain sub and resub
				if len(receivedMessages) % (numberOfEvents/numberOfDrains) == 0 {
					err := msg.Sub.Drain()
					if err != nil {
						t.Fatal(err)
					}
				}
			})
			if err != nil {
				t.Fatal(err)
			}

			timeOfLastSub := time.Now()
			// Block while the sub is valid, waiting for drain to make it invalid
			for subscription.IsValid() {
				// If sub stays alive for a significant amount of time, assume no more messages have
				//  come from nats
				if time.Since(timeOfLastSub) > time.Second * 10 {
					break
				}

				// No CPU thrash please
				time.Sleep(time.Millisecond * 100)
			}


			consumerInfo, err := subscription.ConsumerInfo()
			if err != nil {
				t.Fatal(err)
			}

			// Assuming this is the number the consumer in nats has given to connected clients/subscriptions
			consumerDelivered := consumerInfo.Delivered.Consumer
			// Assuming this is the number the stream has given the consumer on the server
			streamDelivered := consumerInfo.Delivered.Stream
			// Assuming this is a count that has been redelivered to connected clients/subscriptions
			redelivered := consumerInfo.NumRedelivered

			info, err := jsc.StreamInfo(streamName)
			if err != nil {
				t.Fatal(err)
			}
			fmt.Println("Processed messages:", len(receivedMessages), "| Total in Nats:", info.State.Msgs, "| Total Redelivered", redelivered, "| Total Stream Delivered", streamDelivered, "| Total Consumer Delivered", consumerDelivered)

			// If all messages have been processed, break
			if len(receivedMessages) >= numberOfEvents {
				break
			}

			// if draining hasn't happened within 45 seconds, break
			if subscription.IsValid() {
				assert.Equal(t, numberOfEvents, len(receivedMessages))
			}
		}

		fmt.Println("Checking for all messages")
		for i := 0; i < numberOfEvents; i++ {
			if _, ok := messageMap[i+1]; !ok {
				t.Fatal("MISSING MESSAGE", i)
			}
		}
	}

	// Less drains, the issue is less apparent but can still happen
	testNats(t, 10000, 10)
	// More frequent drains seems to cause the issue near 100% of the time
	//  (If this test passes... Please run again)
	testNats(t, 10000, 100)
}

func TestWithDrainOutsideOfCallback(t *testing.T) {
	ctx := context.Background()

	// pass number of events and frequency in seconds of when to drain
	testNats := func(t *testing.T, numberOfEvents int, frequencyOfDrains int) {
		// Starts a nats container to connect to
		connStr := startNatsContainer(ctx)

		// Configure nats and JS client
		jsc := setUpJetStream(t, connStr, streamName, streamSubject, serviceName)

		// Publish events as fast as possible in a go routine, this happens concurrently to the multiple
		//  Subscribes and drains
		go func() {
			for i := 0; i < numberOfEvents; i++ {
				marshal, _ := json.Marshal(map[string]interface{}{
					"number": i+1,
				})

				_, err := jsc.Publish("something.anything", marshal)
				if err != nil {
					panic(err)
				}
			}
		}()

		// Map to verify integrity of events
		messageMap := map[int]*struct{}{}
		// Slice to count number of messages received
		var receivedMessages []*nats.Msg
		// Int to ALSO count number of messages I receive
		msgCount := 0

		// This will never complete because nats appears to be dropping messages despite using .Drain()
		for len(receivedMessages) <= numberOfEvents {
			messagesAtStart := len(receivedMessages)

			subscription, err := jsc.QueueSubscribe("something.>", serviceName, func(msg *nats.Msg) {
				var dataMap map[string]interface{}
				json.Unmarshal(msg.Data, &dataMap)

				// Add and increment data for event
				numberInMessage := int(dataMap["number"].(float64))
				if _, ok := messageMap[numberInMessage]; ok {
					t.Fatal(numberInMessage, "is in list of events more than once")
				}
				messageMap[numberInMessage] = nil

				receivedMessages = append(receivedMessages, msg)

				msgCount++
			})
			if err != nil {
				t.Fatal(err)
			}

			consumerInfo, err := subscription.ConsumerInfo()
			if err != nil {
				t.Fatal(err)
			}

			time.Sleep(time.Second * time.Duration(frequencyOfDrains))
			err = subscription.Drain()
			if err != nil {
				t.Fatal(err)
			}

			for subscription.IsValid() {
				// No CPU thrash please
				time.Sleep(time.Millisecond * 100)
			}

			// Assuming this is the number the consumer in nats has given to connected clients/subscriptions
			consumerDelivered := consumerInfo.Delivered.Consumer
			// Assuming this is the number the stream has given the consumer on the server
			streamDelivered := consumerInfo.Delivered.Stream
			// Assuming this is a count that has been redelivered to connected clients/subscriptions
			redelivered := consumerInfo.NumRedelivered

			info, err := jsc.StreamInfo(streamName)
			if err != nil {
				panic(err)
			}
			fmt.Println("Processed messages:", len(receivedMessages), "| Total in Nats:", info.State.Msgs, "| Total Redelivered", redelivered, "| Total Stream Delivered", streamDelivered, "| Total Consumer Delivered", consumerDelivered)

			// If all messages have been processed, break
			if len(receivedMessages) >= numberOfEvents {
				break
			}

			// If subscription didn't process additional messages, then break as there are no new messages
			if len(receivedMessages) == messagesAtStart {
				break
			}
		}

		fmt.Println("Checking for all messages")
		for i := 0; i < numberOfEvents; i++ {
			if _, ok := messageMap[i+1]; !ok {
				t.Fatal("MISSING MESSAGE", i)
			}
		}
	}

	testNats(t, 100000, 2)
}

func startNatsContainer(ctx context.Context) string {
	port := "4222/tcp"

	cReq := tc.ContainerRequest{
		Image: "nats",
		ExposedPorts: []string{
			port,
		},
		WaitingFor: wait.ForLog("Server is ready"),
		AutoRemove: true,
		Cmd: []string{
			"-js",
		},
	}

	c, err := tc.GenericContainer(ctx, tc.GenericContainerRequest{
		ContainerRequest: cReq,
		Started:          true,
	})
	if err != nil {
		panic(err)
	}

	connStr, err := c.PortEndpoint(ctx, nat.Port(port), "nats")
	if err != nil {
		panic(err)
	}

	return connStr
}

func setUpJetStream(t *testing.T, connStr, streamName, streamSubject, serviceName string) nats.JetStreamContext {
	nc, err := nats.Connect(connStr)
	if err != nil {
		t.Fatal(err)
	}

	jsc, err := nc.JetStream()
	if err != nil {
		t.Fatal(err)
	}

	stream := &nats.StreamConfig{
		Name: streamName,
		Subjects: []string{
			streamSubject,
		},
	}

	_, err = jsc.AddStream(stream)
	if err != nil {
		t.Fatal(err)
	}

	// Add a consumer so the durable connection isn't erased by lib <- ?? Why is this a thing? Defeats the purpose
	//  of durability in my mind...
	_, err = jsc.AddConsumer(streamName, &nats.ConsumerConfig{
		Durable:        serviceName,
		Description:    "A consumer created by the seller service to consume new order events",
		DeliverSubject: strings.Join([]string{serviceName, streamName}, "."),
		DeliverGroup:   serviceName,
		FilterSubject:  "something.>",
		DeliverPolicy: 0,
		OptStartSeq:   0,
		OptStartTime:  nil,
		AckPolicy:     0,
		AckWait:         time.Second * 5,
		MaxDeliver:      0,
		ReplayPolicy:    0,
		RateLimit:       0,
		SampleFrequency: "",
		MaxWaiting:      0,
		MaxAckPending:   0,
		//FlowControl:     true,
		//Heartbeat:       time.Millisecond * 100,
	})
	if err != nil {
		t.Fatal(err)
	}

	return jsc
}