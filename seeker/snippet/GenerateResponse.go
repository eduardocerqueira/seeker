//date: 2022-09-05T17:16:22Z
//url: https://api.github.com/gists/3d85ea4749675c2df3c304495d5f6af8
//owner: https://api.github.com/users/BetterProgramming

func GenerateResponse(Body string, Code int) events.APIGatewayProxyResponse {
 return events.APIGatewayProxyResponse{Body: Body, StatusCode: Code, Headers: map[string]string{
  "Access-Control-Allow-Origin":  "*",
  "Access-Control-Allow-Methods": "POST,OPTIONS",
  "Access-Control-Allow-Headers": "**********"
 }}
}Key,X-Amz-Security-Token,X-Requested-With,X-Auth-Token,Referer,User-Agent,Origin,Content-Type,Authorization,Accept,Access-Control-Allow-Methods,Access-Control-Allow-Origin,Access-Control-Allow-Headers",
 }}
}