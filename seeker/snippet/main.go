//date: 2024-05-30T16:42:46Z
//url: https://api.github.com/gists/02ec5ff78195dbce9e331c217610b4cd
//owner: https://api.github.com/users/alexshd

// This program takes the structured log output and makes it readable.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
)

var (
	reset   = "\033[0m"
	red     = "\033[31m"
	green   = "\033[32m"
	yellow  = "\033[33m"
	blue    = "\033[34;1m"
	magenta = "\033[35m"
	cyan    = "\033[36m"
	gray    = "\033[37m"
	white   = "\033[97;1m"
)

func serviceFlag(service *string) {
	flag.StringVar(service, "service", "", "filter which service to see")
	flag.Parse()
}

func levelIcon(level any) string {
	switch level {
	case "INFO":
		return "✅"

	case "ERROR":
		return "❌"

	default:
		return "😯"
	}

}

func lineFormatter() string {
	// 	  service\   level symbol\ time\ file\           uuid short\                 message\
	return blue + "%s" + reset + " %s %-30s" + cyan + " %-12s" + reset + magenta + " %-.8s" + reset + white + " %-18s" + reset

}
func main() {

	var (
		service string
		b       strings.Builder
	)

	serviceFlag(&service)

	service = strings.ToLower(service)

	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		s := scanner.Text()

		m := make(map[string]any)
		err := json.Unmarshal([]byte(s), &m)
		if err != nil {
			if service == "" {
				fmt.Println(s)
			}
			continue
		}

		// If a service filter was provided, check.
		if service != "" && strings.ToLower(m["service"].(string)) != service {
			continue
		}

		// I like always having a traceid present in the logs.
		traceID := "00000000-0000-0000-0000-000000000000"
		if v, ok := m["trace_id"]; ok {
			traceID = fmt.Sprintf("%v", v)
		}

		var (
			method, addr, path, since, status any
			ok                                bool
		)
		// {"time":"2023-06-01T17:21:11.13704718Z","level":"INFO","msg":"startup","service":"SALES-API","GOMAXPROCS":1}
		if method, ok = m["method"]; !ok {
			method = ""
		}
		// Build out the know portions of the log in the order
		if addr, ok = m["remoteaddr"]; !ok {
			addr = ""
		}
		if path, ok = m["path"]; !ok {
			path = ""
		}
		if since, ok = m["since"]; !ok {
			since = ""
		}
		if status, ok = m["statuscode"]; !ok {
			status = ""
		}
		// I want them in.
		b.Reset()
		b.WriteString(fmt.Sprintf(lineFormatter()+green+"%4v"+reset+yellow+"%4v"+reset+green+" %v "+reset+red+" %v "+reset+green+" %v "+reset,
			m["service"],
			levelIcon(m["level"]),
			m["time"],
			m["file"],
			traceID,
			m["msg"],
			method,
			status,
			path,
			addr,
			since,
		))

		// Add the rest of the keys ignoring the ones we already
		// added for the log.

		// Write the new log format, removing the last :
		out := b.String()
		fmt.Println(out[:len(out)-2])
	}

	if err := scanner.Err(); err != nil {
		log.Println(err)
	}
}
