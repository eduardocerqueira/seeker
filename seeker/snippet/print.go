//date: 2025-05-08T16:58:50Z
//url: https://api.github.com/gists/c554fac2b882ac5c179bffb7d0ad7545
//owner: https://api.github.com/users/mateusoliveira43

import (
	"encoding/json"
	"fmt"
)

// kubernetsObject is a struct, like https://pkg.go.dev/k8s.io/api@v0.31.3/apps/v1#Deployment
formatted, err := json.MarshalIndent(kubernetsObject, "", "\t")
if err != nil {
	// handle error
}
fmt.Printf("FORMATTED\n\n%s\n\n", formatted)
