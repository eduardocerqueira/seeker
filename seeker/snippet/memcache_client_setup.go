//date: 2025-03-05T17:07:34Z
//url: https://api.github.com/gists/79561fe96da7184f2e829646a4bbb3b0
//owner: https://api.github.com/users/kreznicek-mc

package memcache

import (
	"github.com/bradfitz/gomemcache/memcache"
)

// Client is our internal wrapper around the go memcache library
type Client struct {
	Cache  Memcacher
	dialer *tls.Dialer
}

// CreateClient creates a new memcache client.
// As per library doc comment, maxIdleConn should be higher than peak parallel requests.
func CreateClient(maxIdleConn int, endpoints ...string) *Client {
	mem := memcache.New(endpoints...) // single endpoint - serverless memcached read/write
	mem.MaxIdleConns = maxIdleConn // we use 310, can be adjusted

	d := &tls.Dialer{
		Config: &tls.Config{
			MinVersion:         tls.VersionTLS13,
			InsecureSkipVerify: true,
		},
	}
	mem.DialContext = d.DialContext

	return &Client{
		Cache:  mem,
		dialer: d,
	}
}