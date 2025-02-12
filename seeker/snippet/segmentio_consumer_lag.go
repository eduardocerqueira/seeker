//date: 2025-02-12T17:11:26Z
//url: https://api.github.com/gists/ab235f7a34bc1589e5840b11ea8c4fe6
//owner: https://api.github.com/users/MrLis

package main

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/samber/lo"
	"github.com/segmentio/kafka-go"
)

var (
	Brokers = []string{"localhost:9092"}
)

const (
	Cert = `-----BEGIN CERTIFICATE-----
...
-----END CERTIFICATE-----`
	Key = `-----BEGIN RSA PRIVATE KEY-----
...
-----END RSA PRIVATE KEY-----`
	Ca = `-----BEGIN CERTIFICATE-----
...
-----END CERTIFICATE-----`
)

func main() {

	lags, err := getConsumerLag(context.Background())
	if err != nil {
		fmt.Println(err)
	}

	for k, v := range lags {
		fmt.Printf("Topic: %s, ConsumerGroup: %s, Lag: %f\n", k.Topic, k.ConsumerGroup, v)
	}
}

type TopicConsumerGroup struct {
	Topic         string
	ConsumerGroup string
}

func getConsumerLag(ctx context.Context) (map[TopicConsumerGroup]float64, error) {
	res := make(map[TopicConsumerGroup]float64, 0)

	cert, err := tls.X509KeyPair([]byte(Cert), []byte(Key))
	if err != nil {
		return res, fmt.Errorf("failed to parse kafka cert: %w", err)
	}
	certPool := x509.NewCertPool()
	if !certPool.AppendCertsFromPEM([]byte(Ca)) {
		return res, errors.New("failed to parse kafka ca")
	}
	client := &kafka.Client{
		Addr:    kafka.TCP(Brokers...),
		Timeout: 60 * time.Second,
		Transport: &kafka.Transport{
			TLS: &tls.Config{
				Certificates: []tls.Certificate{cert},
				RootCAs:      certPool,
			},
		},
	}

	// get list of topics
	metadata, err := client.Metadata(ctx, &kafka.MetadataRequest{})
	if err != nil {
		return res, fmt.Errorf("failed to request metadata: %w", err)
	}
	allTopics := metadata.Topics

	// get list of consumer groups
	listGroupResp, err := client.ListGroups(ctx, &kafka.ListGroupsRequest{})
	if err != nil {
		return res, fmt.Errorf("failed to request group list: %w", err)
	}

	for _, group := range listGroupResp.Groups {
		// get offsets for each group
		commitedOffsetResp, err := client.OffsetFetch(ctx, &kafka.OffsetFetchRequest{
			GroupID: group.GroupID,
			Topics: lo.SliceToMap(allTopics, func(topic kafka.Topic) (string, []int) {
				return topic.Name, lo.Map(topic.Partitions, func(p kafka.Partition, _ int) int {
					return p.ID
				})
			}),
		})
		if err != nil {
			return res, fmt.Errorf("failed to request commited offsets: %w", err)
		}

		// get last offsets for each topic
		listOffsetsResponse, err := client.ListOffsets(ctx, &kafka.ListOffsetsRequest{
			Topics: lo.SliceToMap(allTopics, func(topic kafka.Topic) (string, []kafka.OffsetRequest) {
				return topic.Name, lo.Map(topic.Partitions, func(p kafka.Partition, _ int) kafka.OffsetRequest {
					return kafka.LastOffsetOf(p.ID)
				})
			}),
		})
		if err != nil {
			return res, fmt.Errorf("failed to request last offsets: %w", err)
		}

		// compute consumer lag
		for topic, partitionOffsets := range listOffsetsResponse.Topics {

			// go through all partitions in topic and compute lag
			for _, partition := range partitionOffsets {
				lastOffset := float64(partition.LastOffset)
				commitedOffset := math.NaN()
				commitedOffsets, ok := commitedOffsetResp.Topics[topic]
				if ok {
					for _, offset := range commitedOffsets {
						if offset.Partition == partition.Partition {
							if offset.CommittedOffset != -1 {
								commitedOffset = float64(offset.CommittedOffset)
							}
						}
					}
				}

				tc := TopicConsumerGroup{
					Topic:         topic,
					ConsumerGroup: group.GroupID,
				}
				if _, ok := res[tc]; !ok {
					res[tc] = 0
				}
				res[tc] = res[tc] + lastOffset - commitedOffset
			}
		}
	}

	return res, nil
}
