//date: 2024-06-06T16:52:21Z
//url: https://api.github.com/gists/e36732a96bde940dc0efefd4d8ac297c
//owner: https://api.github.com/users/jmholzer

package main

import (
    "encoding/json"
    "log"
    "net/http"
    "os"
)

type Response struct {
    Enterprise string `json:"enterprise"`
    APIKey     string `json:"api_key"`
}

func main() {
    http.HandleFunc("/whoami", func(w http.ResponseWriter, r *http.Request) {
        color := os.Getenv("COLOR")
        if color == "" {
            color = "default"
        }

        apiKey := os.Getenv("API_KEY")
        if apiKey == "" {
            apiKey = "default_api_key"
        }

        response := Response{
            Enterprise: color,
            APIKey:     apiKey,
        }
        jsonResponse, err := json.Marshal(response)
        if err != nil {
            http.Error(w, err.Error(), http.StatusInternalServerError)
            return
        }

        w.Header().Set("Content-Type", "application/json")
        w.Write(jsonResponse)
    })

    log.Fatal(http.ListenAndServe(":8080", nil))
}