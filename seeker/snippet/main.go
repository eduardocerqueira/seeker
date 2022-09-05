//date: 2022-09-05T17:07:43Z
//url: https://api.github.com/gists/44b5217461a2e41c60f451393b270c97
//owner: https://api.github.com/users/BetterProgramming

package main
import (
 "context"
"github.com/aws/aws-lambda-go/events"
 "github.com/aws/aws-lambda-go/lambda"
)
func GenerateResponse(Body string, Code int) events.APIGatewayProxyResponse {
 return events.APIGatewayProxyResponse{Body: Body, StatusCode: Code, }
}
func HandleRequest(_ context.Context, request events.LambdaFunctionURLRequest) (events.APIGatewayProxyResponse, error) {
 return GenerateResponse("Hello World", 200), nil
}
func main() {
 lambda.Start(HandleRequest)
}