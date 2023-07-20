//date: 2023-07-20T17:02:14Z
//url: https://api.github.com/gists/a920dcfea6cca057e517443c8a9423c0
//owner: https://api.github.com/users/delveper

package curxrt

import (
	"context"
	"net/http"

	"github.com/GenesisEducationKyiv/main-project-delveper/internal/rate"
	"github.com/GenesisEducationKyiv/main-project-delveper/sys/web"
)

// ExchangeRateHost https://api.exchangerate.host/
type ExchangeRateHost struct{}

func (p ExchangeRateHost) BuildRequest(ctx context.Context, pair rate.CurrencyPair, cfg Config) (*http.Request, error) {
	return newRequest(ctx, cfg.Endpoint,
		web.WithValue("base", pair.Base),
		web.WithValue("symbols", pair.Quote),
	)
}

func (p ExchangeRateHost) ProcessResponse(resp *http.Response) (float64, error) {
	var data struct {
		Motd struct {
			Msg string `json:"msg"`
			URL string `json:"url"`
		} `json:"motd"`
		Success bool               `json:"success"`
		Base    string             `json:"base"`
		Date    string             `json:"date"`
		Rates   map[string]float64 `json:"rates"`
	}

	if err := web.ProcessResponse(resp, &data); err != nil {
		return 0, err
	}

	var val float64

	for _, v := range data.Rates {
		val = v
	}

	return val, nil
}