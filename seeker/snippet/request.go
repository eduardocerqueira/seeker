//date: 2023-07-20T17:02:14Z
//url: https://api.github.com/gists/a920dcfea6cca057e517443c8a9423c0
//owner: https://api.github.com/users/delveper

package web

import (
	"context"
	"net/http"
	"net/url"
	"path"
)

type RequestOption = func(*http.Request)

func ApplyRequestOptions(req *http.Request, opts ...RequestOption) {
	for i := range opts {
		opts[i](req)
	}
}

func WithMethod(meth string) RequestOption {
	return func(req *http.Request) {
		req.Method = meth
	}
}

func WithURL(u *url.URL) RequestOption {
	return func(req *http.Request) {
		req.URL = u
	}
}

func WithValue(key, val string) RequestOption {
	return func(req *http.Request) {
		values := req.URL.Query()
		values.Add(key, val)
		req.URL.RawQuery = values.Encode()
	}
}

func WithAddPath(paths ...string) RequestOption {
	return func(req *http.Request) {
		req.URL.Path = path.Join(append([]string{req.URL.Path}, paths...)...)
	}
}

func WithPath(paths ...string) RequestOption {
	return func(req *http.Request) {
		req.URL.Path = path.Join(paths...)
	}
}

func WithHeader(key, val string) RequestOption {
	return func(req *http.Request) {
		req.Header.Add(key, val)
	}
}

func WithContext(ctx context.Context) RequestOption {
	return func(req *http.Request) {
		*req = *req.WithContext(ctx)
	}
}