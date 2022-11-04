//date: 2022-11-04T17:10:34Z
//url: https://api.github.com/gists/76c1e2ca93dc77cbd0210cfabdeb6dad
//owner: https://api.github.com/users/philipp-zettl

package main


import (
  "bytes"
  "log"
  "io/ioutil"
  "net/url"
  "net/http"
  "net/http/httputil"
  "time"
  "fmt"
)

var new_host = "https://example.com"
var remote = url.URL{}
var current_payload []byte

type ProxyHandler struct {
  p *httputil.ReverseProxy
}

type ResponseLog struct {
  status_code int
  body string
  headers map[string][]string
  // cookies
}

type RequestLog struct {
  payload string

  protocol string
  path string
  method string
  headers map[string][]string
  // cookies
}

type RecordLog struct {
  request RequestLog
  response ResponseLog

}


func dump_response(resp *http.Response, body []byte) {
  req := RecordLog{
    RequestLog{string(current_payload), resp.Request.Proto, resp.Request.URL.String(), resp.Request.Method, resp.Request.Header},
    ResponseLog{resp.StatusCode, string(body), resp.Header},
  }
  current_payload = make([]byte, 0)
  fmt.Printf("%+v\n", req)
}


func parse_response(resp *http.Response) (err error) {
  b, err := ioutil.ReadAll(resp.Body)

  if err != nil {
    panic(err)
    return err
  }
  err = resp.Body.Close()
  if err != nil {
    panic(err)
    return err
  }
  body := ioutil.NopCloser(bytes.NewReader(b))
  resp.Body = body
  dump_response(resp, b)
  return nil

}


func (ph *ProxyHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
  b, err := ioutil.ReadAll(r.Body)

  if err != nil {
    panic(err)
    return
  }
  err = r.Body.Close()
  if err != nil {
    panic(err)
    return
  }
  body := ioutil.NopCloser(bytes.NewReader(b))
  log.Println(fmt.Sprintf("[%s %s] %s", r.Proto, r.Method, r.URL))
  current_payload = b
  r.Body = body
  r.Host = remote.Host
  ph.p.ServeHTTP(w, r)
}

func main() {
  mux := http.NewServeMux()
  remote, err := url.Parse(new_host)
  if err != nil {
    panic(err)
  }

  proxy := httputil.NewSingleHostReverseProxy(remote)
  proxy.ModifyResponse = parse_response
  mux.Handle("/", &ProxyHandler{proxy})
  server := &http.Server{
    Addr:       ":8080",
    Handler:    mux,
    ReadTimeout: 10 * time.Second,
    WriteTimeout: 10 * time.Second,
    MaxHeaderBytes: 1 << 20,
  }
  log.Fatal(server.ListenAndServe())
}