//date: 2022-12-21T17:04:23Z
//url: https://api.github.com/gists/c44cc63e8c0c20bc2b37e8e6b6e4a1a8
//owner: https://api.github.com/users/sjenning

package main

import (
	"bufio"
	"encoding/json"
	"log"
	"os"
	"strings"
)

// https://bitwarden.com/help/condition-bitwarden-import/#condition-a-json

type BitWardenItems struct {
	Items []BitWardenItem `json:"items,omitempty"`
}

const BitWardenLoginType = 1

type BitWardenItem struct {
	Type  int            `json:"type,omitempty"`
	Name  string         `json:"name,omitempty"`
	Login BitWardenLogin `json:"login,omitempty"`
}

type BitWardenLogin struct {
	Password string `json: "**********"
}

func main() {
	// import file is one account per line with account name then password separated by a space
	file, err := os.Open("pass-export.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	var output BitWardenItems

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.Split(scanner.Text(), " ")
		output.Items = append(output.Items, BitWardenItem{
			Type: BitWardenLoginType,
			Name: line[0],
			Login: BitWardenLogin{
				Password: "**********"
			},
		})
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	jsonBytes, err := json.Marshal(output)
	if err != nil {
		log.Fatal(err)
	}

	err = os.WriteFile("bitwarden-import.csv", jsonBytes, 0644)
	if err != nil {
		log.Fatal(err)
	}
}
}
}
