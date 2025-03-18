//date: 2025-03-18T17:01:49Z
//url: https://api.github.com/gists/b6614b3a10fe4118e5a3d015127fed6f
//owner: https://api.github.com/users/kmesiab

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gorilla/mux"
	go_limitless "github.com/kmesiab/go-limitless"
)

// MCPRequest represents a generic MCP message format
type MCPRequest struct {
	Action string                 `json:"action"`
	Params map[string]interface{} `json:"params"`
}

// MCPResponse represents a generic MCP response format
type MCPResponse struct {
	Status  string      `json:"status"`
	Message string      `json:"message,omitempty"`
	Data    interface{} `json:"data,omitempty"`
}

var limitlessClient *go_limitless.Client

func handleMCPRequest(w http.ResponseWriter, r *http.Request) {
	var req MCPRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request payload", http.StatusBadRequest)
		return
	}

	ctx := context.Background()

	var response MCPResponse
	switch req.Action {
	case "get_lifelogs":
		params := parseGetLifelogsParams(req.Params)
		lifelogs, err := limitlessClient.GetLifelogs(ctx, params)
		if err != nil {
			response = MCPResponse{Status: "error", Message: err.Error()}
		} else {
			response = MCPResponse{Status: "success", Data: lifelogs}
		}

	case "get_lifelog":
		lifelogID, ok := req.Params["lifelog_id"].(string)
		if !ok {
			response = MCPResponse{Status: "error", Message: "Missing or invalid lifelog_id"}
		} else {
			lifelog, err := limitlessClient.GetLifelog(ctx, lifelogID)
			if err != nil {
				response = MCPResponse{Status: "error", Message: err.Error()}
			} else {
				response = MCPResponse{Status: "success", Data: lifelog}
			}
		}

	default:
		response = MCPResponse{Status: "error", Message: "Unknown action"}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func parseGetLifelogsParams(params map[string]interface{}) *go_limitless.GetLifelogsParams {
	var p go_limitless.GetLifelogsParams

	if timezone, ok := params["timezone"].(string); ok {
		p.Timezone = timezone
	}
	if date, ok := params["date"].(string); ok {
		p.Date = date
	}
	if start, ok := params["start"].(string); ok {
		parsedStart, err := time.Parse(time.RFC3339, start)
		if err == nil {
			p.Start = parsedStart
		}
	}
	if end, ok := params["end"].(string); ok {
		parsedEnd, err := time.Parse(time.RFC3339, end)
		if err == nil {
			p.End = parsedEnd
		}
	}
	if cursor, ok := params["cursor"].(string); ok {
		p.Cursor = cursor
	}
	if direction, ok := params["direction"].(string); ok {
		p.Direction = direction
	}
	if includeMarkdown, ok := params["includeMarkdown"].(bool); ok {
		p.IncludeMarkdown = &includeMarkdown
	}
	if includeHeadings, ok := params["includeHeadings"].(bool); ok {
		p.IncludeHeadings = &includeHeadings
	}
	if limit, ok := params["limit"].(float64); ok {
		p.Limit = int(limit)
	}

	return &p
}

func main() {
	apiKey := os.Getenv("LIMITLESS_API_KEY")
	if apiKey == "" {
		log.Fatal("LIMITLESS_API_KEY environment variable is required")
	}

	limitlessClient = go_limitless.NewClient(apiKey)

	r := mux.NewRouter()
	r.HandleFunc("/mcp", handleMCPRequest).Methods("POST")

	log.Println("MCP server is running on port 8080...")
	http.ListenAndServe(":8080", r)
}