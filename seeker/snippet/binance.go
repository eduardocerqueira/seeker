//date: 2025-02-04T16:38:50Z
//url: https://api.github.com/gists/ab7740a433cea07014522b892e77da5a
//owner: https://api.github.com/users/DefaultPerson

package scrappers

import (
	"bytes"
	"context"
	"encoding/json"
	"github.com/rs/zerolog/log"
	"io"
	"net/http"
	"test/utils"
	"time"
)

type binanceAnnounce struct {
	ID          int    `json:"id"`
	Code        string `json:"code"`
	Title       string `json:"title"`
	ReleaseDate int64  `json:"releaseDate"`
}

type binanceAnnouncesResponse struct {
	Data struct {
		Catalogs []struct {
			Articles []binanceAnnounce `json:"articles"`
		} `json:"catalogs"`
	} `json:"data"`
}

type BinanceExtended interface {
	Scrapper
	FilterAnnouncement(text string)
}

type BinanceScrappy struct {
	BaseScrappy
}

func (bs BinanceScrappy) IsProxyError(
	transport *http.Transport,
	body []byte,
	cfg *utils.Statistic,
) {
	if bytes.Contains(body, []byte("Cloudflare")) ||
		bytes.Contains(body, []byte("cf-error-details")) {
		cfg.BadProxies++
	}
}
func (bs BinanceScrappy) FilterAnnouncement(
	text string,
) {

}
func (bs BinanceScrappy) DeserializeData(
	body []byte,
	schema struct{},
) {
	// Если хотим что-то вернуть, придётся менять интерфейс,
	// но сейчас просто обрабатываем внутри
	if bytes.Contains(body, []byte("Cloudflare")) ||
		bytes.Contains(body, []byte("cf-error-details")) {
		log.Info().Msg("Cloudflare error inside!")
	}
}

func (bs *BinanceScrappy) Run(
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
	bs.run(
		bs,
		ctx,
		channels,
		wsHub,
		pm,
		telegramRetryDelay,
		telegramRetryCount,
		statsReportPeriod,
		proxyCooldown,
		announcesCount,
		delay,
		domainCount,
	)
}

func (bs BinanceScrappy) MakeRequest(
	//b BinanceScrapper,
	client *http.Client,
	pm utils.ProxyManager,
	ctx context.Context,
	channels []utils.Channel,
	wsHub *utils.Hub,
	telegramRetryDelay int,
	telegramRetryCount int,
	announcesCount int,
	stats *utils.Statistic,
) {
	for {
		transport := pm.GetTransport()
		if transport == nil {
			log.Warn().Msg("No available proxies, waiting...")
			time.Sleep(5 * time.Second)
			break
		}
		client.Transport = transport
		resp, err := client.Get(bs.targetUrl)
		stats.TotalRequests++

		if err != nil {
			//if proxyChecker, ok := b.(BinanceScrappy); ok {
			log.Info().Msg("Detected IsProxyError method, executing...")
			bs.IsProxyError(nil, nil, stats)
			//}
		}
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return
			//return nil, fmt.Errorf("read body failed: %w", err)
		}
		var response binanceAnnouncesResponse
		json.Unmarshal(body, &response)
		log.Info().Msgf("Response: %v", response)
		break
	}

}

func NewBinanceScrapper() BinanceExtended {
	bs := &BinanceScrappy{
		BaseScrappy: BaseScrappy{
			exchange:  "binance",
			targetUrl: "https://www.binance.com/bapi/apex/v1/public/apex/cms/article/list/query?type=1&pageNo=1&pageSize=10&catalogId=48",
		},
	}
	var extended BinanceExtended = bs
	return extended
}
