//date: 2021-12-22T17:05:06Z
//url: https://api.github.com/gists/c76215e28153a79e53d49e26d6879a0f
//owner: https://api.github.com/users/nicolai86

package main

import (
	"fmt"
	"github.com/hashicorp/terraform-config-inspect/tfconfig"
	"log"
	"os"
	"strings"
)

func checkForViolations(dir string) ([]string, error) {
	module, diags := tfconfig.LoadModule(dir)
	if diags.HasErrors() {
		return nil, diags.Err()
	}
	violations := []string{}
	for name, module := range module.ModuleCalls {
		nested, err := checkForViolations(dir + "/" + module.Source)
		if err != nil {
			return nil, fmt.Errorf("failed checking module %s: %w", name, err)
		}
		for _, violation := range nested {
			violations = append(violations, fmt.Sprintf("module.%s.%s", name, violation))
		}
	}
	for name, resource := range module.ManagedResources {
		if strings.Contains(name, "-") {
			violations = append(violations, fmt.Sprintf("%s.%s", resource.Type, resource.Name))
		}
	}
	return violations, nil
}

func main() {
	dir := os.Args[1]
	violations, err := checkForViolations(dir)
	if err != nil {
		log.Fatalf("failed to check for violations: %s\n", err)
	}
	for _, violation := range violations {
		fmt.Printf("%s is not snake_cased\n", violation)
	}
	if len(violations) > 0 {
		os.Exit(1)
	}
}
