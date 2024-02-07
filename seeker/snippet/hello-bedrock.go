//date: 2024-02-07T16:57:28Z
//url: https://api.github.com/gists/fe9401be7ba252223d8fe1844b08f8bb
//owner: https://api.github.com/users/micahwalter

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

func main() {

	cfg, err := config.LoadDefaultConfig(context.TODO(), config.WithRegion("us-east-1"))
	if err != nil {
		log.Fatalf("unable to load AWS SDK config, %v", err)
	}

	svc := bedrockruntime.NewFromConfig(cfg)

	accept := "*/*"
	contentType := "application/json"
	modelId := "anthropic.claude-v2:1"

	prompt : "**********":\"\\n\\nHuman: Write a short poem about autumn\\n\\nAssistant:\",\"max_tokens_to_sample\":300,\"temperature\":1,\"top_k\":250,\"top_p\":0.999,\"stop_sequences\":[\"\\n\\nHuman:\"],\"anthropic_version\":\"bedrock-2023-05-31\"}"

	resp, err := svc.InvokeModel(context.TODO(), &bedrockruntime.InvokeModelInput{
		Accept:      &accept,
		ModelId:     &modelId,
		ContentType: &contentType,
		Body:        []byte(string(prompt)),
	})
	if err != nil {
		log.Fatalf("error from Bedrock, %v", err)
	}

	var out map[string]any

	err = json.Unmarshal(resp.Body, &out)
	if err != nil {
		fmt.Printf("unable to Unmarshal JSON, %v", err)
	}

	fmt.Print(out["completion"])

}}