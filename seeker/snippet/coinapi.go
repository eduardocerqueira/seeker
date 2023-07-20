//date: 2023-07-20T17:02:14Z
//url: https://api.github.com/gists/a920dcfea6cca057e517443c8a9423c0
//owner: https://api.github.com/users/delveper

package curxrt

import (
	"context"
	"net/http"
	"time"

	"github.com/GenesisEducationKyiv/main-project-delveper/internal/rate"
	"github.com/GenesisEducationKyiv/main-project-delveper/sys/web"
)

// CoinApi https://docs.coinapi.io/market-data/rest-api/exchange-rates
type CoinApi struct{}

func (p CoinApi) BuildRequest(ctx context.Context, pair rate.CurrencyPair, cfg Config) (*http.Request, error) {
	return newRequest(ctx, cfg.Endpoint,
		web.WithAddPath(pair.Base, pair.Quote),
		web.WithHeader(cfg.Header, cfg.Key),
	)
}

func (p CoinApi) ProcessResponse(resp *http.Response) (float64, error) {
	var data struct {
		Time         time.Time `json:"time"`
		AssetIdBase  string    `json:"asset_id_base"`
		AssetIdQuote string    `json:"asset_id_quote"`
		Rate         float64   `json:"rate"`
	}

	if err := web.ProcessResponse(resp, &data); err != nil {
		return 0, err
	}

	return data.Rate, nil
}