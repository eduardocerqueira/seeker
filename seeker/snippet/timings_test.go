//date: 2022-12-01T17:11:39Z
//url: https://api.github.com/gists/17b682b71ddb7ee9f999d9d5dae4dbbc
//owner: https://api.github.com/users/jeromelaurens

package util

import (
	"context"
	"github.com/ethereum/go-ethereum"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/ethclient"
	"github.com/samber/lo"
	log "github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sync"
	"testing"
	"time"
)

// TestTimingsNewHead compare timings of new heads  between different EL  aka geth, erigon ...
func TestTimingsNewHead(t *testing.T) {
	ctx := context.Background()

	type headerTiming struct {
		name     string
		dateTime time.Time
		block    int64
	}
	timingForAllNodes := make(chan []headerTiming)
	defer close(timingForAllNodes)
	mu := &sync.RWMutex{}

	checkTimings := func(expectedTimings int, blockNumber int64, headerTimings map[int64][]headerTiming) bool {
		mu.RLock()
		defer mu.RUnlock()
		if len(headerTimings[blockNumber]) >= expectedTimings {
			return true
		}
		return false
	}
	headerTimingsAccumulator := make(map[int64][]headerTiming)
	processHeaderTimings := func(name, url string, expectedTimings int) (headerChan chan *types.Header, sub ethereum.Subscription) {
		ethClient, err := ethclient.Dial(url)
		require.NoError(t, err)
		headerChan = make(chan *types.Header)
		sub, errSub := ethClient.SubscribeNewHead(ctx, headerChan)
		require.NoError(t, errSub)
		go func(headerChan chan *types.Header, ethClientName string) {
			for header := range headerChan {
				block := header.Number.Int64()
				mu.Lock()
				currentTimings := headerTimingsAccumulator[block]
				headerTimingsAccumulator[header.Number.Int64()] = append(currentTimings, headerTiming{
					name:     ethClientName,
					dateTime: time.Now(),
					block:    header.Number.Int64(),
				})
				mu.Unlock()
				complete := checkTimings(expectedTimings, block, headerTimingsAccumulator)
				mu.RLock()
				timing := headerTimingsAccumulator[block]
				mu.RUnlock()
				if complete {
					timingForAllNodes <- timing
				}
			}
		}(headerChan, name)
		return
	}

	type ethClient struct {
		name string
		url  string
	}
	ethClients := make([]ethClient, 3)

	ethClients[0] = ethClient{
		name: "geth_lighthouse    ", //space so they are aligned in logs
		url:  <your_ws_url>,
	}
	ethClients[1] = ethClient{
		name: "erigon_lighthouse  ", //space so they are aligned in logs
		url:  <your_ws_url>,
	}
	ethClients[2] = ethClient{
		name: "erigon_finedTuned  ", //space so they are aligned in logs
		url:  <your_ws_url>,
	}

	for _, client := range ethClients {
		chanToClose, sub := processHeaderTimings(client.name, client.url, len(ethClients))
		//closing properly everything
		defer func() {
			sub.Unsubscribe()
			close(chanToClose)
		}()
	}

	const countBlocksForAvg = 10
	const expectedMaxAvgMs = 50
	stats := make(map[string][]int64)

	avgFunc := func(rawDiffs []int64, name string) int {
		sum := lo.Reduce[int64, int64](rawDiffs, func(agg int64, item int64, _ int) int64 {
			return agg + item
		}, 0)
		return int(sum) / len(rawDiffs)
	}
	i := 0
	for timings := range timingForAllNodes {
		firstTime := timings[0].dateTime
		for _, timing := range timings {
			logFields := log.Fields{
				"block":  timing.block,
				"from":   timing.name,
				"timing": timing.dateTime.Format("2006-01-02T15:04:05.999"),
			}
			diff := timing.dateTime.Sub(firstTime).Milliseconds()
			logFields["y_diffMs"] = diff // y to have them at the end in logs
			stats[timing.name] = append(stats[timing.name], diff)
			avg := avgFunc(stats[timing.name], timing.name)
			logFields["z_avg"] = avg // z to have them at the end in logs
			log.WithFields(logFields).Info("got timing")
		}
		log.Println("")
		i++
		if i >= countBlocksForAvg {
			break
		}
	}

	for clientName, int64s := range stats {
		avg := avgFunc(int64s, clientName)
		assert.Less(t, avg, expectedMaxAvgMs, "%s too slow for newHeads", clientName)
	}

}
