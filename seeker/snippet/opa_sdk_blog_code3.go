//date: 2021-09-09T17:13:21Z
//url: https://api.github.com/gists/3aff7d9cdc2d5b5bc303728aad7fc420
//owner: https://api.github.com/users/TheLunaticScripter

func main() {
  ...

  decisionOptions := sdk.DecisionOptions{
		// Rule to be called example /rules/allow
		Path: "/rules/allow",
		// Input to be used in the decision example {"user": "alice"}
		Input: map[string]interface{}{
			"user": "bob",
			"method": "GET",
		},
	}
}
