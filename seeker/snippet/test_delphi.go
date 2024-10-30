//date: 2024-10-30T17:12:10Z
//url: https://api.github.com/gists/5fb84f31a889acd7ec26175b05f9f19e
//owner: https://api.github.com/users/etai-shuchatowitz

package main

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"time"

	mlmodelv1 "go.viam.com/api/service/mlmodel/v1"
	"google.golang.org/api/idtoken"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	grpcMetadata "google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/types/known/structpb"
)

// NewConn creates a new gRPC connection.
// host should be of the form domain:port, e.g., example.com:443
func NewConn(host string, secure bool) (*grpc.ClientConn, error) {
	var opts []grpc.DialOption
	if host != "" {
		opts = append(opts, grpc.WithAuthority(host))
	}

	if !secure {
		opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	} else {
		// Note: On the Windows platform, use of x509.SystemCertPool() requires
		// go version 1.18 or higher.
		systemRoots, err := x509.SystemCertPool()
		if err != nil {
			return nil, err
		}
		cred := credentials.NewTLS(&tls.Config{
			RootCAs: systemRoots,
		})
		opts = append(opts, grpc.WithTransportCredentials(cred))
	}

	return grpc.NewClient(host, opts...)
}

// callDelphiWithAuth mints a new Identity Token for each request.
// This token has a 1 hour expiry and should be reused.
// audience must be the auto-assigned URL of a Cloud Run service or HTTP Cloud Function without port number.
func callDelphiWithAuth(conn *grpc.ClientConn, req *mlmodelv1.InferRequest, audience string) (*mlmodelv1.InferResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
	defer cancel()

	// Create an identity token.
	// With a global TokenSource tokens would be reused and auto-refreshed at need.
	// A given TokenSource is specific to the audience.
	tokenSource, err : "**********"
	if err != nil {
		return nil, fmt.Errorf("idtoken.NewTokenSource: "**********"
	}
	token, err : "**********"
	if err != nil {
		return nil, fmt.Errorf("TokenSource.Token: "**********"
	}

	// Add token to gRPC Request.
	ctx = "**********"

	fmt.Printf("\n110: "**********": %s\n", "**********".AccessToken)

	// Send the request.
	client := mlmodelv1.NewMLModelServiceClient(conn)
	res, err := client.Infer(ctx, req)
	fmt.Printf("\nres is: %s\n", res)
	return res, err
}

func main() {
	fmt.Print("I am starting")
	host := "inference-service-bplesliplq-uc.a.run.app:443"
	conn, err := NewConn(host, true)
	if err != nil {
		fmt.Printf("77 err: %s", err)
		return
	}
	// fmt.Printf("\n79 conn: %s\n", conn)
	audience := "https://inference-service-1025988320057.us-central1.run.app"
	req := &mlmodelv1.InferRequest{
		Extra: &structpb.Struct{
			Fields: map[string]*structpb.Value{
				"image_url": &structpb.Value{
					Kind: &structpb.Value_StringValue{
						StringValue: "hello world",
					},
				},
			},
		},
	}
	res, err := callDelphiWithAuth(conn, req, audience)
	if err != nil {
		fmt.Printf("\n97 err: %s\n", err)
	}
	fmt.Printf("\n96: %s\n", res)
	conn.Close()
}
 {
		fmt.Printf("\n97 err: %s\n", err)
	}
	fmt.Printf("\n96: %s\n", res)
	conn.Close()
}
