//date: 2023-03-15T16:48:42Z
//url: https://api.github.com/gists/ecfd9c3c5a427dddb1e7850ed6f113ce
//owner: https://api.github.com/users/DylanGuedes

package loki

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grafana/dskit/flagext"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/common/model"
	"github.com/prometheus/prometheus/model/labels"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/weaveworks/common/server"

	"github.com/grafana/loki/pkg/chunkenc"
	ingesterclient "github.com/grafana/loki/pkg/ingester/client"
	"github.com/grafana/loki/pkg/logproto"
	internalserver "github.com/grafana/loki/pkg/server"
	"github.com/grafana/loki/pkg/storage"
	"github.com/grafana/loki/pkg/storage/chunk"
	"github.com/grafana/loki/pkg/util/cfg"
	util_log "github.com/grafana/loki/pkg/util/log"
	"github.com/grafana/loki/pkg/validation"
)

func TestGenerateMyData(t *testing.T) {
	var defaultsConfig Config

	if err := cfg.Unmarshal(&defaultsConfig, cfg.Defaults(flag.CommandLine)); err != nil {
		fmt.Println("Failed parsing defaults config:", err)
		os.Exit(1)
	}

	var lokiCfg ConfigWrapper
	destArgs := []string{"-config.file=../../cmd/loki/loki-local-config.yaml"}
	if err := cfg.DynamicUnmarshal(&lokiCfg, destArgs, flag.NewFlagSet("config-file-loader", flag.ContinueOnError)); err != nil {
		fmt.Fprintf(os.Stderr, "failed parsing config: %v\n", err)
		os.Exit(1)
	}

	if err := lokiCfg.Validate(); err != nil {
		fmt.Println("Failed to validate dest store config:", err)
		os.Exit(1)
	}

	limits, err := validation.NewOverrides(lokiCfg.LimitsConfig, nil)
	if err != nil {
		fmt.Println("Failed to create limit overrides:", err)
		os.Exit(1)
	}

	// Create a new registerer to avoid registering duplicate metrics
	prometheus.DefaultRegisterer = prometheus.NewRegistry()
	clientMetrics := storage.NewClientMetrics()
	store, err := storage.NewStore(lokiCfg.StorageConfig, lokiCfg.ChunkStoreConfig, lokiCfg.SchemaConfig, limits, clientMetrics, prometheus.DefaultRegisterer, util_log.Logger)
	if err != nil {
		fmt.Println("Failed to create store:", err)
		os.Exit(1)
	}

	oneDay := 24 * time.Hour
	oneDayAgo := model.Now().Add(-oneDay)
	thirtyDaysAgo := model.Now().Add(-30 * oneDay)
	halfYearAgo := model.Now().Add(-180 * oneDay)

	c1 := createChunk(t, "org1", labels.Labels{labels.Label{Name: "foo", Value: "bar"}},
		oneDayAgo, oneDayAgo.Add(time.Hour), func() string { return "1 day ago" })

	c2 := createChunk(t, "org1", labels.Labels{labels.Label{Name: "foo", Value: "buzz"}, labels.Label{Name: "bar", Value: "foo"}},
		thirtyDaysAgo, thirtyDaysAgo.Add(time.Hour), func() string { return "30 days ago" })

	c3 := createChunk(t, "org1", labels.Labels{labels.Label{Name: "foo", Value: "buzz"}, labels.Label{Name: "bar", Value: "foo"}},
		halfYearAgo, halfYearAgo.Add(time.Hour), func() string { return "180 days ago" })
	require.NoError(t, store.Put(context.TODO(), []chunk.Chunk{
		c1, c2, c3,
	}))
}

func createChunk(t testing.TB, userID string, lbs labels.Labels, from model.Time, through model.Time, lineBuilder func() string) chunk.Chunk {
	t.Helper()
	const (
		targetSize = 1500 * 1024
		blockSize  = 256 * 1024
	)
	labelsBuilder := labels.NewBuilder(lbs)
	labelsBuilder.Set(labels.MetricName, "logs")
	metric := labelsBuilder.Labels(nil)
	fp := ingesterclient.Fingerprint(lbs)
	chunkEnc := chunkenc.NewMemChunk(chunkenc.EncSnappy, chunkenc.UnorderedHeadBlockFmt, blockSize, targetSize)

	for ts := from; !ts.After(through); ts = ts.Add(1 * time.Minute) {
		lineContent := ts.String()
		if lineBuilder != nil {
			lineContent = lineBuilder()
		}
		require.NoError(t, chunkEnc.Append(&logproto.Entry{
			Timestamp: ts.Time(),
			Line:      lineContent,
		}))
	}

	require.NoError(t, chunkEnc.Close())
	c := chunk.NewChunk(userID, fp, metric, chunkenc.NewFacade(chunkEnc, blockSize, targetSize), from, through)
	require.NoError(t, c.Encode())
	return c
}
