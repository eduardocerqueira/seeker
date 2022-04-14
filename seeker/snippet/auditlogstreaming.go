//date: 2022-04-14T16:52:40Z
//url: https://api.github.com/gists/5d17dbbf6ee7f03a17120c637a3e539a
//owner: https://api.github.com/users/salrashid123

package main

import (
	"fmt"
	"io"
	"os"

	logging "cloud.google.com/go/logging/apiv2"
	"golang.org/x/net/context"
	"google.golang.org/genproto/googleapis/cloud/audit"
	loggingpb "google.golang.org/genproto/googleapis/logging/v2"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
)

const ()

var ()

func main() {

	projectID := "fabled-ray-104117"

	ctx := context.Background()

	c, err := logging.NewClient(ctx)
	if err != nil {
		fmt.Printf("%v", err)
		os.Exit(-1)
	}
	defer c.Close()
	stream, err := c.TailLogEntries(ctx)
	if err != nil {
		fmt.Printf("%v", err)
		os.Exit(-1)
	}
	go func() {
		reqs := []*loggingpb.TailLogEntriesRequest{{
			ResourceNames: []string{fmt.Sprintf("projects/%s", projectID)},
		}}
		for _, req := range reqs {
			if err := stream.Send(req); err != nil {
				fmt.Printf("%v", err)
				os.Exit(-1)
			}
		}
		//stream.CloseSend()
	}()
	for {
		resp, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			fmt.Printf("%v", err)
			os.Exit(-1)
		}

		for _, e := range resp.Entries {
			fmt.Printf("Entry: %v\n", e.InsertId)
			var audit_data audit.AuditLog

			err := anypb.UnmarshalTo(e.GetProtoPayload(), &audit_data, proto.UnmarshalOptions{})
			if err != nil {
				fmt.Printf("Error: could not unmarshall to audit log")
			} else {
				fmt.Printf("   Authentication: %v\n", audit_data.AuthenticationInfo)
				fmt.Printf("   Authorization: %v\n", audit_data.AuthorizationInfo)
			}
		}
	}

}
