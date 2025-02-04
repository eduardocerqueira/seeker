//date: 2025-02-04T16:38:50Z
//url: https://api.github.com/gists/ab7740a433cea07014522b892e77da5a
//owner: https://api.github.com/users/DefaultPerson

package scrappers

import (
	"context"
	"github.com/rs/zerolog/log"
	"net/http"
	"test/utils"
	"time"
)

const RequestTimeout = 5

type Scrapper interface {
	Run(
		ctx context.Context,
		channels []utils.Channel,
		wsHub *utils.Hub,
		pm utils.ProxyManager,
		telegramRetryDelay int,
		telegramRetryCount int,
		statsReportPeriod int,
		proxyCooldown int,
		announcesCount int,
		delay float64,
		domainCount int,
	)
	MakeRequest(
		client *http.Client,
		pm utils.ProxyManager,
		ctx context.Context,
		channels []utils.Channel,
		wsHub *utils.Hub,
		telegramRetryDelay int,
		telegramRetryCount int,
		announcesCount int,
		stats *utils.Statistic,
	)
	IsProxyError(
		transport *http.Transport,
		body []byte,
		cfg *utils.Statistic,
	)
	DeserializeData(
		body []byte,
		schema struct{},
	)
}

type BaseScrappy struct {
	exchange  string
	targetUrl string
}

func (bs *BaseScrappy) run(
	s Scrapper,
	ctx context.Context,
	channels []utils.Channel,
	wsHub *utils.Hub,
	pm utils.ProxyManager,
	telegramRetryDelay int,
	telegramRetryCount int,
	statsReportPeriod int,
	proxyCooldown int,
	announcesCount int,
	delay float64,
	domainCount int,
) {
	calcDelay := time.Duration(delay) * time.Second
	time.Sleep(calcDelay)
	log.Info().Msgf("Scrapper started with delay %s", calcDelay)
	if s == nil {
		panic("self is nil! Ensure you assign self before calling methods")
	}

	stats := utils.Statistic{}
	statsTicker := time.Second * time.Duration(statsReportPeriod)
	go utils.StatsReporter(statsTicker, bs.exchange, &stats)
	go pm.CheckRestoreTransports()

	client := &http.Client{
		Timeout: RequestTimeout * time.Second,
	}
	for {
		go s.MakeRequest(
			client, pm, ctx, channels, wsHub,
			telegramRetryDelay, telegramRetryCount, announcesCount, &stats,
		)
		stats.BadProxies = pm.CountBad()
		stats.ActiveProxies = pm.Count()

		interval := time.Duration(
			float64(time.Second) / (float64(pm.Count()) / float64(proxyCooldown)) * float64(domainCount),
		)
		stats.CurrentDuration = interval
		time.Sleep(interval)
	}
}
