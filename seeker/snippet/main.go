//date: 2024-03-01T16:54:52Z
//url: https://api.github.com/gists/4a44f573112d6d1cebb456850534e40d
//owner: https://api.github.com/users/thtg88

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/dynamodb"
	"github.com/aws/aws-sdk-go/service/dynamodb/dynamodbattribute"
)

type DynamoOuterItem struct {
	Item DynamoItem
}

type DynamoItem struct {
	Authors         map[string][]string
	Dimensions      map[string]string
	ISBN            map[string]string
	Id              map[string]string
	InPublication   map[string]bool
	PageCount       map[string]string
	Price           map[string]string
	ProductCategory map[string]string
	Title           map[string]string
}

type Item struct {
	Authors         []string
	Dimensions      string
	ISBN            string
	Id              int64
	InPublication   bool
	PageCount       int64
	Price           int64
	ProductCategory string
	Title           string
}

const fileName = "items-small.json"

func main() {
	// profile execution
	executionProfile, err := os.Create(fmt.Sprintf("profiles/executionprofile-%s", time.Now()))
	if err != nil {
		log.Fatal("could not create trace execution profile: ", err)
	}
	defer executionProfile.Close()
	trace.Start(executionProfile)
	defer trace.Stop()

	file, err := os.Open(fileName)
	if err != nil {
		log.Fatalf("error while opening the file %s: %v", fileName, err)
	}
	defer file.Close()

	decoder := json.NewDecoder(file)

	// while the array contains values
	for decoder.More() {
		var doi DynamoOuterItem
		// decode an array value (Message)
		err := decoder.Decode(&doi)
		if err != nil {
			log.Printf("could not decode value: %v", err)
			continue
		}

		avMap := map[string]*dynamodb.AttributeValue{
			"Authors": {
				SS: aws.StringSlice(doi.Item.Authors["SS"]),
			},
			"Dimensions": {
				S: aws.String(doi.Item.Dimensions["S"]),
			},
			"ISBN": {
				S: aws.String(doi.Item.ISBN["S"]),
			},
			"Id": {
				N: aws.String(doi.Item.Id["N"]),
			},
			"InPublication": {
				BOOL: aws.Bool(doi.Item.InPublication["B"]),
			},
			"PageCount": {
				N: aws.String(doi.Item.PageCount["N"]),
			},
			"Price": {
				N: aws.String(doi.Item.Price["N"]),
			},
			"ProductCategory": {
				S: aws.String(doi.Item.ProductCategory["S"]),
			},
			"Title": {
				S: aws.String(doi.Item.Title["S"]),
			},
		}

		var item Item
		err = dynamodbattribute.UnmarshalMap(avMap, &item)
		if err != nil {
			log.Printf("could not unmarshal map: %v", err)
			return
		}

		log.Printf("value parsed: %v", item)
	}

	// profile memory usage
	memoryProfile, err := os.Create(fmt.Sprintf("profiles/memprofile-%s", time.Now()))
	if err != nil {
		log.Fatal("could not create memory profile: ", err)
	}
	defer memoryProfile.Close()
	runtime.GC()
	if err := pprof.WriteHeapProfile(memoryProfile); err != nil {
		log.Fatal("could not write memory profile: ", err)
	}
}
