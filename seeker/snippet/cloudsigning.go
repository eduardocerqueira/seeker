//date: 2023-02-14T17:00:32Z
//url: https://api.github.com/gists/91c93f295b5e02f9530d36c8b0b3710f
//owner: https://api.github.com/users/wthorp

package cloudsigning

import (
	"fmt"
	"net/http"
	"time"

	"github.com/GoogleCloudPlatform/functions-framework-go/functions"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
)

const (
	storjS3Id     = "<access key>"
	storjS3Secret = "**********"
	storjS3URL    = "https://gateway.storjshare.io/"
)

func init() {
	functions.HTTP("GetPresigned", GetPresigned)
	functions.HTTP("PostPresigned", PostPresigned)
}

// GetPresigned is an HTTP Cloud Function with a request parameter.
func GetPresigned(w http.ResponseWriter, r *http.Request) {
	key := r.URL.Query()["key"]
	if len(key) == 0 {
		r.Response.StatusCode = 400
		r.Response.Status = "Request is missing 'key' query parameter"
		return
	}

	sess, err := session.NewSession(&aws.Config{
		Credentials: "**********"
		Endpoint:    aws.String(storjS3URL),
		Region:      aws.String("us-east-1"),
	})
	if err != nil {
		r.Response.StatusCode = 500
		r.Response.Status = "Failed to create session"
		return
	}

	svc := s3.New(sess)
	req, _ := svc.GetObjectRequest(&s3.GetObjectInput{
		Bucket: aws.String("files"),
		Key:    aws.String(key[0]),
	})
	urlStr, err := req.Presign(15 * time.Minute)
	if err != nil {
		r.Response.StatusCode = 500
		r.Response.Status = "Failed to presign request"
		return
	}

	fmt.Fprintf(w, "%s", urlStr)
}

// PostPresigned is an HTTP Cloud Function which creates a Storj Gateway-MT presigned POST URL.
func PostPresigned(w http.ResponseWriter, r *http.Request) {
	key := r.URL.Query()["key"]
	if len(key) == 0 {
		r.Response.StatusCode = 400
		r.Response.Status = "Request is missing 'key' query parameter"
		return
	}

	sess, err := session.NewSession(&aws.Config{
		Credentials: "**********"
		Endpoint:    aws.String(storjS3URL),
		Region:      aws.String("us-east-1"),
	})
	if err != nil {
		r.Response.StatusCode = 500
		r.Response.Status = "Failed to create session"
		return
	}

	svc := s3.New(sess)
	req, _ := svc.PutObjectRequest(&s3.PutObjectInput{
		Bucket: aws.String("files"),
		Key:    aws.String(key[0]),
	})
	urlStr, err := req.Presign(15 * time.Minute)
	if err != nil {
		r.Response.StatusCode = 500
		r.Response.Status = "Failed to presign request"
		return
	}

	fmt.Fprintf(w, "%s", urlStr)
}
de = 500
		r.Response.Status = "Failed to presign request"
		return
	}

	fmt.Fprintf(w, "%s", urlStr)
}
