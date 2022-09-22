//date: 2022-09-22T17:19:44Z
//url: https://api.github.com/gists/0f941373da4b59b51c0150b4f7c35613
//owner: https://api.github.com/users/jniltinho

package main

// go mod init get-jira-id
// gofmt -w main.go
// go build -ldflags="-s -w"
// upx --best --lzma get-jira-id
// ./get-jira-id --issue=feature/Pbo-784-epic

import (
	"flag"
	"os"
	"regexp"
	"strings"
)

func main() {

	issueID := flag.String("issue", "feature/PBO-5599-epic", "Issue ID name")
	flag.Parse()
	GetIssueID(*issueID)
	//GetIssueID("fc/PBO-559999-epic-NNNN-VVVGGG")
}

func GetIssueID(issueIDName string) {
	re := regexp.MustCompile(`(\b(pbo|asd|asdp|ae|at))+-[\d]+`)
	matches := re.FindAllString(strings.ToLower(issueIDName), -1)

	if len(matches) == 0 {
		print("not_found_jira_id")
		os.Exit(1)
	}
	print(matches[0])
	os.Exit(0)
}
