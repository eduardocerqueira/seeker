//date: 2023-05-23T17:04:33Z
//url: https://api.github.com/gists/0d08516c3e7cc2bd48e92e88502eeea7
//owner: https://api.github.com/users/destag

package combahton

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/spf13/viper"
	"github.com/valyala/fasthttp"
)

const (
	baseURL = "https://api.combahton.net"
)

var client *Client

type Client struct {
	baseURL    string
	httpClient *fasthttp.Client
	username   string
	password   string
}

type Response[T any] struct {
	Status struct {
		Code  int    `json:"code"`
		Type  string `json:"type"`
		Error bool   `json:"error"`
	} `json:"status"`
	Paginated bool `json:"paginated,omitempty"`
	Count     int  `json:"count,omitempty"`
	PerPage   int  `json:"per_page,omitempty"`
	Page      int  `json:"page,omitempty"`
	Result    T    `json:"result"`
}

func NewClient() *Client {
	return &Client{
		baseURL:    baseURL,
		httpClient: &fasthttp.Client{},
		username:   viper.GetString("app.ddosusername"),
		password: "**********"
	}
}

func Init() {
	client = NewClient()
}

func (c *Client) get(endpoint string) ([]byte, error) {
	url := c.baseURL + endpoint

	req := fasthttp.AcquireRequest()
	req.SetRequestURI(url)
	req.Header.SetMethod(fasthttp.MethodGet)
	req.Header.Set("Accept", "application/json")
	req.Header.Set(
		"Authorization",
		"Basic "+base64.StdEncoding.EncodeToString([]byte(c.username+": "**********"
	)

	resp := fasthttp.AcquireResponse()
	err := c.httpClient.Do(req, resp)
	fasthttp.ReleaseRequest(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %s", err)
	}
	defer fasthttp.ReleaseResponse(resp)

	if resp.StatusCode() != http.StatusOK {
		return resp.Body(), fmt.Errorf("request failed with status code: %d", resp.StatusCode())
	}

	return resp.Body(), nil
}

func (c *Client) post(endpoint string, payload interface{}) ([]byte, error) {
	url := c.baseURL + endpoint

	data, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal JSON: %s", err)
	}

	req := fasthttp.AcquireRequest()
	req.SetRequestURI(url)
	req.Header.SetMethod(fasthttp.MethodPost)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set(
		"Authorization",
		"Basic "+base64.StdEncoding.EncodeToString([]byte(c.username+": "**********"
	)
	req.SetBody(data)

	resp := fasthttp.AcquireResponse()
	err = c.httpClient.Do(req, resp)
	fasthttp.ReleaseRequest(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %s", err)
	}
	defer fasthttp.ReleaseResponse(resp)

	if resp.StatusCode() != http.StatusOK {
		return resp.Body(), fmt.Errorf("request failed with status code: %d", resp.StatusCode())
	}

	return resp.Body(), nil
}

func (c *Client) delete(endpoint string) ([]byte, error) {
	url := c.baseURL + endpoint

	req := fasthttp.AcquireRequest()
	req.SetRequestURI(url)
	req.Header.SetMethod(fasthttp.MethodDelete)
	req.Header.Set("Accept", "application/json")
	req.Header.Set(
		"Authorization",
		"Basic "+base64.StdEncoding.EncodeToString([]byte(c.username+": "**********"
	)

	resp := fasthttp.AcquireResponse()
	err := c.httpClient.Do(req, resp)
	fasthttp.ReleaseRequest(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %s", err)
	}
	defer fasthttp.ReleaseResponse(resp)

	if resp.StatusCode() != http.StatusOK {
		return resp.Body(), fmt.Errorf("request failed with status code: %d", resp.StatusCode())
	}

	return resp.Body(), nil
}

	}

	return resp.Body(), nil
}
