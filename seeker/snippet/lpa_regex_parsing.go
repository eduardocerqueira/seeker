//date: 2024-06-24T16:49:03Z
//url: https://api.github.com/gists/8ba321a549523c7b1ceafe9da85df808
//owner: https://api.github.com/users/ik5

package main

// go play with my original code: https://go.dev/play/p/dt8WFHM1E8d

import (
	"fmt"
	"regexp"
)

var validLPA = []string{
	`LPA$www1.a.google.com$ABC1`,
	`LPA$www1.a.google.com$ABC1-X$1.1.1$1`,
	`LPA$www1.a.google.com$ABC1$$1`,
	`LPA$www1.a.google.com$ABC1$10000.2.3333333`,
}

const activationCodeRegex = "**********"

var (
	activationCodeRegexp = regexp.MustCompile(activationCodeRegex)
	groupNames           = activationCodeRegexp.SubexpNames()
)

func main() {
	fmt.Printf("groupNames: %+#v\n", groupNames)
	var (
		matches [][]string
		result  []map[string]string
	)
	for _, lpaString := range validLPA {
		matches = activationCodeRegexp.FindAllStringSubmatch(lpaString, -1)
		fmt.Printf("%s - %#+v\n", lpaString, matches)

		var tmpMap = map[string]string{}
		for idx, name := range groupNames {
			if name != "" && idx > 0 {
				tmpMap[name] = matches[0][idx]
			}
		}
		result = append(result, tmpMap)

	}

	fmt.Printf("result: %+#v\n", result)
}p = map[string]string{}
		for idx, name := range groupNames {
			if name != "" && idx > 0 {
				tmpMap[name] = matches[0][idx]
			}
		}
		result = append(result, tmpMap)

	}

	fmt.Printf("result: %+#v\n", result)
}