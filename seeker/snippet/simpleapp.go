//date: 2023-12-13T17:05:31Z
//url: https://api.github.com/gists/8d8e8b1242b82cc6b30974dd7f0aad48
//owner: https://api.github.com/users/antonok

// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// [START bigquery_simple_app_all]

// Command simpleapp queries the Stack Overflow public dataset in Google BigQuery.
package main

// [START bigquery_simple_app_deps]
import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"reflect"

	"cloud.google.com/go/bigquery"
	"google.golang.org/api/iterator"
)

// [END bigquery_simple_app_deps]

func main() {
	projectID := os.Getenv("GOOGLE_CLOUD_PROJECT")
	if projectID == "" {
		fmt.Println("GOOGLE_CLOUD_PROJECT environment variable must be set.")
		os.Exit(1)
	}

	// [START bigquery_simple_app_client]
	ctx := context.Background()

	client, err := bigquery.NewClient(ctx, projectID)
	if err != nil {
		log.Fatalf("bigquery.NewClient: %v", err)
	}
	defer client.Close()
	// [END bigquery_simple_app_client]

	rows, err := query(ctx, client)
	if err != nil {
		log.Fatal(err)
	}
	//if err := printResults(os.Stdout, rows); err != nil {
	//	log.Fatal(err)
	//}
	var soRows []StackOverflowRow
	if err := readRowsWithReflect(os.Stdout, rows, &soRows); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%v", soRows)
}

func readRows(w io.Writer, iter *bigquery.RowIterator, rows *[]StackOverflowRow) error {
	for {
		var row StackOverflowRow
		err := iter.Next(&row)
		if err == iterator.Done {
			return nil
		}
		if err != nil {
			return fmt.Errorf("error iterating through results: %w", err)
		}
		*rows = append(*rows, row)
	}
	return nil
}

func readRowsWithReflect(w io.Writer, iter *bigquery.RowIterator, rows interface{}) error {
	pRows := reflect.ValueOf(rows).Elem()
	for {
		rowValue := newRow(rows).Elem()
		row := rowValue.Interface()//.(StackOverflowRow)
		rowPtr := &row
		rowPtrType := reflect.TypeOf(rowPtr)
		fmt.Printf("are you a pointer? %s\n", rowPtrType.Kind())
		fmt.Printf("are you a struct? %s\n", rowPtrType.Elem().Kind())

		err := iter.Next(rowPtr)
		if err == iterator.Done {
			return nil
		}
		if err != nil {
			return fmt.Errorf("error iterating through results: %w", err)
		}
		fmt.Printf("%v\n", row)
		pRows.Set(reflect.Append(pRows, rowValue))
	}
	return nil
}

func newRow(rows interface{}) reflect.Value {
	//get the type of variable (array pointer), then array and type of array
	rowType := reflect.TypeOf(rows).Elem().Elem()
	fmt.Printf("value %v\n", rowType)
	//pRowType := reflect.PtrTo(rowType)
	//fmt.Printf("* value %v\n", pRowType)
	//return reflect.New(pRowType)
	return reflect.New(rowType)
//	return reflect.New(reflect.PtrTo(rowType))
}

// query returns a row iterator suitable for reading query results.
func query(ctx context.Context, client *bigquery.Client) (*bigquery.RowIterator, error) {

	// [START bigquery_simple_app_query]
	query := client.Query(
		`SELECT
			CONCAT(
				'https://stackoverflow.com/questions/',
				CAST(id as STRING)) as url,
			view_count
		FROM ` + "`bigquery-public-data.stackoverflow.posts_questions`" + `
		WHERE tags like '%google-bigquery%'
		ORDER BY view_count DESC
		LIMIT 10;`)
	return query.Read(ctx)
	// [END bigquery_simple_app_query]
}

// [START bigquery_simple_app_print]
type StackOverflowRow struct {
	URL       string `bigquery:"url"`
	ViewCount int64  `bigquery:"view_count"`
}

// printResults prints results from a query to the Stack Overflow public dataset.
func printResults(w io.Writer, iter *bigquery.RowIterator) error {
	for {
		var row StackOverflowRow
		err := iter.Next(&row)
		if err == iterator.Done {
			return nil
		}
		if err != nil {
			return fmt.Errorf("error iterating through results: %w", err)
		}

		fmt.Fprintf(w, "url: %s views: %d\n", row.URL, row.ViewCount)
	}
}

// [END bigquery_simple_app_print]
// [END bigquery_simple_app_all]