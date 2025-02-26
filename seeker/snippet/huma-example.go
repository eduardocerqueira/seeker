//date: 2025-02-26T16:52:39Z
//url: https://api.github.com/gists/358540fca96d8b76c30221be04aa7d32
//owner: https://api.github.com/users/DLzer

package main

import (
	"context"
	"fmt"
	"net/http"

	"github.com/danielgtaylor/huma/v2"
	"github.com/danielgtaylor/huma/v2/adapters/humachi"
	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"

	_ "github.com/danielgtaylor/huma/v2/formats/cbor"
)

const (
	ServerName    = "test-api-1"
	ServerVersion = "1.5.5"
)

// StatusOutput represents the API status response.
type StatusOutput struct {
	Body struct {
		Status  string `json:"status" example:"OK" doc:"Status Message"`
		Code    int32  `json:"code" example:"200" doc:"Status Code"`
		Version string `json:"version" example:"v1.2.3" doc:"Status Version"`
	}
}

// PingOutput represents the PingPong response handler
type PingOutput struct {
	Body struct {
		Pong string `json:"pong" example:"pong" doc:"Ping Pong Response"`
	}
}

// UserOutput represents the User response output
type UserOutput struct {
	Body struct {
		ID   int32  `json:"id" example:"315" doc:"User ID"`
		Name string `json:"name" example:"John Smith" doc:"User Name"`
	}
}

func main() {
	// Create a new router & API
	router := chi.NewMux()
	api := humachi.New(router, huma.DefaultConfig(ServerName, ServerVersion))

	// Register GET /status
	huma.Register(api, huma.Operation{
		OperationID: "status",
		Method:      http.MethodGet,
		Path:        "/status",
		Summary:     "Get the server status",
		Description: "Returns a simple status body.",
		Tags:        []string{"Status"},
	}, func(ctx context.Context, input *struct{}) (*StatusOutput, error) {
		resp := &StatusOutput{}
		resp.Body.Code = 200
		resp.Body.Status = "OK"
		resp.Body.Version = ServerVersion
		return resp, nil
	})

	// Register GET /ping/{msg}
	huma.Register(api, huma.Operation{
		OperationID: "ping",
		Method:      http.MethodGet,
		Path:        "/ping/{pong}",
		Summary:     "Return the ping message",
		Description: "Accepts a ping message and returns it as a pong.",
		Tags:        []string{"Ping"},
	}, func(ctx context.Context, input *struct {
		Pong string `path:"pong" maxLength:"30" example:"Hello, world" doc:"A simple message"`
	}) (*PingOutput, error) {
		resp := &PingOutput{}
		resp.Body.Pong = fmt.Sprintf("Hello, %s", input.Pong)
		return resp, nil
	})

	// Register a new /v1 Huma Group
	grp := huma.NewGroup(api, "/v1")

	// Register a new sub /users Huma Group
	usersGrp := huma.NewGroup(grp, "/users")
	// The /users group utilizes it's own huma-centric middleware
	usersGrp.UseMiddleware(RequestIdMiddleware)

	huma.Get(usersGrp, "/", func(ctx context.Context, input *struct{}) (*UserOutput, error) {
		resp := &UserOutput{}
		resp.Body.ID = int32(515)
		resp.Body.Name = "John Smith"
		return resp, nil
	}, func(o *huma.Operation) {
		o.Summary = "Get user information"
		o.Description = "Returns user details including ID and name"
		o.Tags = []string{"Users"}
		o.OperationID = "getUserInfo"
	},
	)

	// Start the server!
	http.ListenAndServe("127.0.0.1:8888", router)
}

// RequestIdMiddleware appends a request ID using the Huma Middleware huma.Context
func RequestIdMiddleware(ctx huma.Context, next func(huma.Context)) {
	rid := uuid.New().String()

	// Interal
	huma.WithValue(ctx, "request_id", rid)

	// External
	ctx.AppendHeader("x-request-id", rid)

	next(ctx)
}