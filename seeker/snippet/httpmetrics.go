//date: 2025-09-25T16:47:35Z
//url: https://api.github.com/gists/db4ff2080f082f4ea3c1c633cc2da9bf
//owner: https://api.github.com/users/brbarmex

// go:build go1.20
package httpmetrics

import (
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/DataDog/datadog-go/v5/statsd"
)

type PathNormalizer func(host, rawPath string) string

type RT struct {
	Base           http.RoundTripper
	Stats          *statsd.Client
	BaseTags       []string
	MaxPathLen     int
	NormalizePath  PathNormalizer
	MetricNS       string // ex: "myapp." (terminar com ponto) ou "" se não quiser namespace
}

// New cria um RoundTripper que mede métricas HTTP e Cosmos.
// - base: transporte base (http.Transport). Se nil, usa http.DefaultTransport.
// - stats: cliente DogStatsD já conectado ao Agent.
// - baseTags: tags fixas (ex.: service:api, env:prod).
func New(base http.RoundTripper, stats *statsd.Client, baseTags []string) *RT {
	if base == nil {
		base = http.DefaultTransport
	}
	return &RT{
		Base:       base,
		Stats:      stats,
		BaseTags:   baseTags,
		MaxPathLen: 96,
		MetricNS:   "", // opcional, pode deixar vazio e usar namespace no statsd.New
		NormalizePath: func(host, p string) string {
			// padrão seguro: só retorna o path sem IDs; pode customizar via setter
			if p == "" {
				return "/"
			}
			return p
		},
	}
}

func (m *RT) RoundTrip(req *http.Request) (*http.Response, error) {
	start := time.Now()

	method := strings.ToLower(req.Method)
	host := strings.ToLower(req.URL.Hostname())
	path := req.URL.EscapedPath()
	path = m.NormalizePath(host, path)
	if len(path) > m.MaxPathLen {
		path = path[:m.MaxPathLen] + "…"
	}

	// request size (se houver)
	reqSize := parseFloat64(req.Header.Get("Content-Length"))

	resp, err := m.Base.RoundTrip(req)

	elapsed := time.Since(start)
	status := 0
	var hdr http.Header
	if resp != nil {
		status = resp.StatusCode
		hdr = resp.Header
	}

	// tags comuns
	tags := append([]string{
		"http.method:" + method,
		"http.status_code:" + itoa(status),
		"http.host:" + host,
		"http.path:" + path,
		"result:" + resultTag(err, status),
	}, m.BaseTags...)

	// ——— extras Cosmos ———
	ru := parseFloat64(hdr.Get("x-ms-request-charge"))
	retryAfterMs := parseFloat64(hdr.Get("x-ms-retry-after-ms"))
	if sub := hdr.Get("x-ms-substatus"); sub != "" {
		tags = append(tags, "cosmos.substatus:"+sub)
	}
	if act := hdr.Get("x-ms-activity-id"); act != "" {
		tags = append(tags, "cosmos.activity_id:"+act)
	}

	ns := m.MetricNS
	// métricas principais http
	_ = m.Stats.Timing(ns+"http.client.request.latency", elapsed, tags, 1)
	_ = m.Stats.Count(ns+"http.client.request.count", 1, tags, 1)

	// bytes
	if reqSize >= 0 {
		_ = m.Stats.Gauge(ns+"http.client.request.bytes", reqSize, tags, 1)
	}
	if resp != nil && resp.ContentLength >= 0 {
		_ = m.Stats.Gauge(ns+"http.client.response.bytes", float64(resp.ContentLength), tags, 1)
	}

	// erros
	if err != nil || status >= 500 || status == 0 {
		_ = m.Stats.Count(ns+"http.client.request.errors", 1, tags, 1)
	}

	// cosmos: RU e throttling
	if ru >= 0 {
		_ = m.Stats.Gauge(ns+"cosmos.request.ru", ru, tags, 1)
		_ = m.Stats.Distribution(ns+"cosmos.request.ru.dist", ru, tags, 1)
	}
	if status == http.StatusTooManyRequests {
		_ = m.Stats.Count(ns+"cosmos.request.throttled", 1, tags, 1)
		if retryAfterMs >= 0 {
			_ = m.Stats.Gauge(ns+"cosmos.request.retry_after_ms", retryAfterMs, tags, 1)
		}
	}

	return resp, err
}

// —— helpers ——
func parseFloat64(s string) float64 {
	if s == "" {
		return -1
	}
	// aceitar vírgula decimal
	if strings.Contains(s, ",") && !strings.Contains(s, ".") {
		s = strings.ReplaceAll(s, ",", ".")
	}
	f, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return -1
	}
	return f
}
func itoa(i int) string { return strconv.Itoa(i) }
func resultTag(err error, code int) string {
	if err != nil || code == 0 {
		return "error"
	}
	if code >= 400 {
		return "fail"
	}
	return "ok"
}


============

package main

import (
	"context"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/data/azcosmos"
	"github.com/DataDog/datadog-go/v5/statsd"
	ddhttp "github.com/DataDog/dd-trace-go/v2/contrib/net/http"
	"github.com/DataDog/dd-trace-go/v2/ddtrace"
	"github.com/DataDog/dd-trace-go/v2/ddtrace/tracer"

	httpmetrics "seu/modulo/httpmetrics"
)

func main() {
	// ——— APM ———
	tracer.Start(
		tracer.WithService("api"),
		tracer.WithEnv("prod"),
		tracer.WithRuntimeMetrics(),
	)
	defer tracer.Stop()

	// ——— StatsD ———
	stats, _ := statsd.New("127.0.0.1:8125",
		statsd.WithNamespace("myapp."), // opcional: prefixo nas métricas
		statsd.WithTags([]string{"service:api", "env:prod"}),
	)

	// ——— Transporte base (pool e timeouts “padrão cloud”) ———
	base := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   5 * time.Second,
			KeepAlive: 60 * time.Second,
		}).DialContext,
		MaxIdleConns:        200,
		MaxIdleConnsPerHost: 100,
		IdleConnTimeout:     90 * time.Second,
		TLSHandshakeTimeout: 5 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		ForceAttemptHTTP2:   true,
	}

	// ——— Métricas (StatsD) ———
	metricsRT := httpmetrics.New(base, stats, []string{"component:httpclient"})
	metricsRT.MetricNS = "" // já usamos namespace no cliente StatsD acima
	metricsRT.NormalizePath = normalizeCosmosPath // normalizador p/ reduzir cardinalidade

	// ——— APM (tracing) ———
	client := ddhttp.WrapClient(&http.Client{
		Transport: metricsRT,        // <— métricas + cosmos headers
		Timeout:   30 * time.Second, // deadline total
	},
		ddhttp.RTWithServiceName("cosmos-http"),
		ddhttp.RTWithSpanName("http.client"),
		ddhttp.RTWithResourceNamer(func(r *http.Request) string {
			// resource “curto e estável” (bom pro APM)
			return r.Method + " " + normalizeCosmosPath(r.URL.Path)
		}),
		ddhttp.RTWithSpanModifier(func(sp ddtrace.Span, r *http.Request, resp *http.Response, err error) {
			// enriquecer span com headers Cosmos (se existir)
			if resp == nil {
				return
			}
			h := resp.Header
			if v := h.Get("x-ms-request-charge"); v != "" {
				sp.SetTag("cosmos.request_charge", v)
			}
			if v := h.Get("x-ms-substatus"); v != "" {
				sp.SetTag("cosmos.substatus", v)
			}
			if v := h.Get("x-ms-retry-after-ms"); v != "" {
				sp.SetTag("cosmos.retry_after_ms", v)
			}
			if v := h.Get("x-ms-activity-id"); v != "" {
				sp.SetTag("cosmos.activity_id", v)
			}
		}),
	)

	// ——— CosmosDB client usando o http.Client instrumentado ———
	cred, err := azcosmos.NewKeyCredential("<COSMOS_KEY>")
	if err != nil {
		panic(err)
	}
	endpoint := "https://<sua-conta>.documents.azure.com:443/"

	cosmos, err := azcosmos.NewClientWithKey(endpoint, cred, &azcosmos.ClientOptions{
		ClientOptions: azcore.ClientOptions{
			Transport: client, // <— todas as requisições passam por métricas + tracing
		},
	})
	if err != nil {
		panic(err)
	}

	// ——— Exemplo de operação: ReadItem ———
	ctx := context.Background()
	_ = cosmos // use ReadItem/Query/UpsertItem/DeleteItem normalmente — métricas e spans saem automáticas
	_ = ctx
}

// normalizeCosmosPath reduz cardinalidade trocando IDs por placeholders.
// Ex.: /dbs/mydb/colls/orders/docs/123 -> cosmos:/dbs/:db/colls/:coll/docs
func normalizeCosmosPath(p string) string {
	p = strings.ToLower(p)
	p = strings.ReplaceAll(p, "/documents", "/docs")
	parts := strings.Split(p, "/")
	out := make([]string, 0, len(parts))
	for i := 0; i < len(parts); i++ {
		s := parts[i]
		switch s {
		case "dbs":
			out = append(out, "dbs"); i++; out = append(out, ":db")
		case "colls":
			out = append(out, "colls"); i++; out = append(out, ":coll")
		case "docs":
			out = append(out, "docs")
		default:
			// ignore segmentos variáveis para evitar cardinalidade (IDs, guids, números)
			// se quiser manter algumas rotas, trate caso a caso aqui.
		}
	}
	if len(out) == 0 {
		return "/"
	}
	return "cosmos:/" + strings.Join(out, "/")
}

