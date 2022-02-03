//date: 2022-02-03T17:08:00Z
//url: https://api.github.com/gists/c128157608f86c5f1be4a12b19ab71d0
//owner: https://api.github.com/users/nickistre

package utils

import (
	"regexp"
	"strings"
)

// SnakeCaseToCamelCase
// This will convert snake case to camelCased
// @return camelCased string, number of parts
func SnakeCaseToCamelCase(inputUnderScoreStr string) (string, int) {
	parts := strings.Split(inputUnderScoreStr, "_")
	for index := range parts {
		if index != 0 {
			parts[index] = strings.Title(strings.ToLower(parts[index]))
		} else {
			parts[index] = strings.ToLower(parts[index])
		}
	}
	return strings.Join(parts, ""), len(parts)
}

// CamelCaseToSnakeCase
// This will convert camelCase to snake_cased
// @return snakeCased string, number of parts
func CamelCaseToSnakeCase(inputCamelCaseStr string) (string, int) {
	// Regex from https://www.golangprograms.com/split-a-string-at-uppercase-letters-using-regular-expression-in-golang.html
	re := regexp.MustCompile(`[A-z][^A-Z]*`)
	parts := re.FindAllString(inputCamelCaseStr, -1)
	for index := range parts {
		parts[index] = strings.ToLower(parts[index])
	}
	return strings.Join(parts, "_"), len(parts)
}
