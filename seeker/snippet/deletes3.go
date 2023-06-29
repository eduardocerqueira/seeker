//date: 2023-06-29T16:51:40Z
//url: https://api.github.com/gists/310d04777bb26a403248138cf5ee73e8
//owner: https://api.github.com/users/brunoksato

package main

import (
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
)

func exitErrorf(msg string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, msg+"\n", args...)
}

type S3Deleter struct {
	client *s3.S3
	bucket string
	mfa    string
	quiet  bool
}

func NewS3Deleter(bucket string, quiet bool, debug bool) (*S3Deleter, error) {
	sess := session.Must(session.NewSession())

	var loglevel *aws.LogLevelType
	if debug {
		loglevel = aws.LogLevel(aws.LogDebugWithRequestRetries)
	}
	cfg := &aws.Config{
		Region: aws.String("us-east-1"),
		HTTPClient: &http.Client{
			Timeout: time.Minute,
		},
		MaxRetries: aws.Int(0),
		LogLevel:   loglevel,
	}
	d := &S3Deleter{
		client: s3.New(sess, cfg),
		bucket: bucket,
		quiet:  quiet,
	}
	return d, nil
}

func (d *S3Deleter) DeleteKeys(keys []string) error {
	del := &s3.Delete{
		Objects: toOIDs(keys),
		Quiet:   &d.quiet,
	}
	doi := &s3.DeleteObjectsInput{
		Bucket: &d.bucket,
		Delete: del,
	}
	res, err := d.client.DeleteObjects(doi)
	if err != nil {
		return err
	}
	if len(res.Errors) != 0 {
		return d.newBatchError(res.Errors)
	}
	return nil
}

func (_ *S3Deleter) newBatchError(errs []*s3.Error) error {
	msgs := make([]string, len(errs))
	for i := 0; i < len(errs); i += 1 {
		msgs[i] = *errs[i].Message
	}
	return NewBatchError(msgs)
}

type BatchError struct {
	error
	Messages []string
}

func NewBatchError(msgs []string) error {
	if len(msgs) == 0 {
		panic("msgs must contain one or more messages")
	}
	msg := msgs[0]
	if more := len(msgs) - 1; more > 0 {
		msg += fmt.Sprintf(" (and %d more errors)", more)
	}
	return BatchError{errors.New(msg), msgs}
}

func toOIDs(keys []string) []*s3.ObjectIdentifier {
	ret := make([]*s3.ObjectIdentifier, len(keys))
	for i := 0; i < len(ret); i += 1 {
		oid := &s3.ObjectIdentifier{
			Key: &(keys[i]),
		}
		ret[i] = oid
	}
	return ret
}

func main() {
	bucket := "prod.assets.onquidd.com"

	var s3Deleter *S3Deleter
	s3Deleter, mainErr := NewS3Deleter(bucket, true, false)
	if mainErr != nil {
		fmt.Fprintln(os.Stderr, mainErr.Error())
		os.Exit(1)
	}

	totalIterated := 0
	input := &s3.ListObjectsV2Input{
		Bucket: aws.String(bucket),
		Prefix: aws.String("shelfies/uploads"),
	}

	total := 0
	pageNum := 0
	keys := []string{}
	err := s3Deleter.client.ListObjectsV2Pages(input, func(page *s3.ListObjectsV2Output, lastPage bool) bool {
		pageNum++

		for _, item := range page.Contents {
			if strings.Contains(*item.Key, "-share.jpg") || strings.Contains(*item.Key, "-share.png") || strings.Contains(*item.Key, "@w") {
				// objects = append(objects, s3manager.BatchDeleteObject{
				// 	Object: &s3.DeleteObjectInput{
				// 		Key:    item.Key,
				// 		Bucket: aws.String(bucket),
				// 	},
				// })
				keys = append(keys, *item.Key)
				// fmt.Println("Deleting", *item.Key)
			}
		}

		if len(keys) >= 1000 {
			start := time.Now()
			err := s3Deleter.DeleteKeys(keys)
			if err != nil {
				fmt.Println("Error deleting keys", err)
			}
			total += len(keys)
			elapsed := time.Since(start)
			fmt.Println("Total deleted ", total, " took: ", elapsed)
		}

		// if len(objects) >= 1000 {
		// 	start := time.Now()
		// 	batcher := s3manager.NewBatchDelete(sess, func(bd *s3manager.BatchDelete) {
		// 		bd.BatchSize = 1000
		// 	})
		// 	if err := batcher.Delete(aws.BackgroundContext(), &s3manager.DeleteObjectsIterator{
		// 		Objects: objects,
		// 	}); err != nil {
		// 		exitErrorf("Unable to delete items in bucket %q, %v", bucket, err)
		// 	}
		// 	total += len(objects)
		// 	elapsed := time.Since(start)
		// 	fmt.Println("Total deleted ", total, "items from bucket", bucket, " took: ", elapsed)
		// 	objects = []s3manager.BatchDeleteObject{}
		// }

		totalIterated += len(page.Contents)
		// fmt.Println("Total iterated", totalIterated, "items from bucket", bucket)

		return !lastPage
	})
	if err != nil {
		exitErrorf("Unable to list items in bucket %q, %v", bucket, err)
	}

}
