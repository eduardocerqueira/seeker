//date: 2022-09-05T17:15:29Z
//url: https://api.github.com/gists/708bbd4a5a4019037462ccdf5c9c81cb
//owner: https://api.github.com/users/BetterProgramming

...
func HandleRequest(_ context.Context, request events.LambdaFunctionURLRequest) (events.APIGatewayProxyResponse, error) {
 if SetupError != nil {
  // We can query our database as follows:
  // defer db.Close()
  // results, err := db.Query("SELECT id, name FROM tags")
  return GenerateResponse("Hello World", 200), nil
 } else {
  // throw error if our database connection fails
  return GenerateResponse("Error connecting to DB", 500), nil
 }
}
...