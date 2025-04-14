//date: 2025-04-14T16:36:37Z
//url: https://api.github.com/gists/08e4acccb7dace269c1b344c5101e7cd
//owner: https://api.github.com/users/vizee

package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"strings"
)

type GithubRepo struct {
	CloneUrl      string `json:"clone_url"`
	DefaultBranch string `json:"default_branch"`
	Description   string `json:"description"`
	FullName      string `json:"full_name"`
	Name          string `json:"name"`
	SshUrl        string `json:"ssh_url"`
}

type GiteeRepo struct {
	SshUrl string      `json:"ssh_url"`
	Error  *GiteeError `json:"error"`
}

type GiteeError struct {
	Base []string `json:"base"`
}

var (
	githubApiToken = "**********"
	giteeApiToken  = "**********"
)

func listGhUserRepo(username string, page int, perPage int) ([]*GithubRepo, error) {
	/*
		curl -L \
		  -H "Accept: application/vnd.github+json" \
		  -H "Authorization: "**********"
		  -H "X-GitHub-Api-Version: 2022-11-28" \
		  https://api.github.com/users/USERNAME/repos?per_page=2&page=2
	*/
	hreq, err := http.NewRequest(http.MethodGet, fmt.Sprintf("https://api.github.com/users/%s/repos?per_page=%d&page=%d", username, perPage, page), nil)
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Accept", "application/vnd.github+json")
	hreq.Header.Set("Authorization", "Bearer "+githubApiToken)
	hreq.Header.Set("X-GitHub-Api-Version", "2022-11-28")
	resp, err := http.DefaultClient.Do(hreq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("github api error: %s", resp.Status)
	}
	respData, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var repos []*GithubRepo
	err = json.Unmarshal(respData, &repos)
	if err != nil {
		return nil, err
	}
	return repos, nil
}

func listGhOrgRepo(orgname string, page int, perPage int) ([]*GithubRepo, error) {
	/*
		curl -L \
		  -H "Accept: application/vnd.github+json" \
		  -H "Authorization: "**********"
		  -H "X-GitHub-Api-Version: 2022-11-28" \
		  https://api.github.com/orgs/ORG/repos?per_page=2&page=2
	*/
	hreq, err := http.NewRequest(http.MethodGet, fmt.Sprintf("https://api.github.com/orgs/%s/repos?per_page=%d&page=%d", orgname, perPage, page), nil)
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Accept", "application/vnd.github+json")
	hreq.Header.Set("Authorization", "Bearer "+githubApiToken)
	hreq.Header.Set("X-GitHub-Api-Version", "2022-11-28")
	resp, err := http.DefaultClient.Do(hreq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("github api error: %s", resp.Status)
	}
	respData, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var repos []*GithubRepo
	err = json.Unmarshal(respData, &repos)
	if err != nil {
		return nil, err
	}
	return repos, nil
}

var (
	dryRun = false
)

func createGiteeRepo(ghRepo *GithubRepo, suffix string) (*GiteeRepo, error) {
	log.Printf("create gitee repo: %s", ghRepo.Name)
	if dryRun {
		return &GiteeRepo{
			SshUrl: strings.Replace(ghRepo.SshUrl, "github.com", "gitee.com", 1),
		}, nil
	}
	reqData, err := json.Marshal(map[string]any{
		"access_token": "**********"
		"name":         ghRepo.Name + suffix,
		"description":  ghRepo.Description,
		"private":      "true",
		// default_branch
		"default_branch": ghRepo.DefaultBranch,
		"auto_init":      "false",
		// has_issues
		"has_issues": "false",
		// has_page
		"has_page": "false",
		// has_wiki
		"has_wiki": "false",
		// can_comment
		"can_comment": "false",
	})
	if err != nil {
		return nil, err
	}
	resp, err := http.Post("https://gitee.com/api/v5/user/repos", "application/json", bytes.NewReader(reqData))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("gitee api error: %s", resp.Status)
	}
	respData, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var repo GiteeRepo
	err = json.Unmarshal(respData, &repo)
	if err != nil {
		return nil, err
	}
	if repo.Error != nil {
		return nil, fmt.Errorf("gitee api error: %s", strings.Join(repo.Error.Base, "\n"))
	}
	return &repo, nil
}

func runCommand(wd string, bin string, args ...string) error {
	if dryRun {
		fmt.Println("+", bin, strings.Join(args, " "), "["+wd+"]")
		return nil
	} else {
		cmd := exec.Command(bin, args...)
		if wd != "" {
			cmd.Dir = wd
		}
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		return cmd.Run()
	}
}

func migrateRepo(ghRepo *GithubRepo, suffix string) error {
	err := os.MkdirAll("/tmp/repos", 0755)
	if err != nil {
		return err
	}
	name := ghRepo.Name
	if name == "..." {
		name = "archives"
	}
	repoDir := fmt.Sprintf("/tmp/repos/%s", name)
	// git clone github-repo to /tmp/repos
	err = runCommand("", "git", "clone", ghRepo.SshUrl, repoDir)
	if err != nil {
		return err
	}

	gtRepo, err := createGiteeRepo(ghRepo, suffix)
	if err != nil {
		return err
	}

	// git remote set origin to gitee
	err = runCommand(repoDir, "git", "remote", "set-url", "origin", gtRepo.SshUrl)
	if err != nil {
		return err
	}
	// git push --all
	return runCommand(repoDir, "git", "push", "--all")
}

func main() {
	var (
		org      bool
		startIdx int
		perPage  int
		page     int
		suffix   string
	)
	flag.BoolVar(&org, "org", false, "migrate org repo")
	flag.IntVar(&startIdx, "start", 0, "start index")
	flag.IntVar(&perPage, "per-page", 100, "per page size")
	flag.IntVar(&page, "page", 1, "page")
	flag.BoolVar(&dryRun, "dry-run", false, "dry run")
	flag.StringVar(&suffix, "suffix", "", "suffix for gitee repo name")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintf(os.Stderr, "Usage: %s [options] <username|orgname>\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}

	var ghRepos []*GithubRepo
	if org {
		repos, err := listGhOrgRepo(flag.Arg(0), page, perPage)
		if err != nil {
			log.Fatalf("listGhOrgRepo: %v\n", err)
		}
		ghRepos = repos
	} else {
		repos, err := listGhUserRepo(flag.Arg(0), page, perPage)
		if err != nil {
			log.Fatalf("listGhUserRepo: %v\n", err)
		}
		ghRepos = repos
	}
	if len(ghRepos) == 0 {
		log.Printf("no repo found")
		return
	}
	for i, ghRepo := range ghRepos[startIdx:] {
		if strings.HasSuffix(ghRepo.Name, ".github.io") {
			continue
		}
		log.Printf("migrating [%d] %s", startIdx+i, ghRepo.FullName)
		err := migrateRepo(ghRepo, suffix)
		if err != nil {
			log.Fatalf("migrateRepo: %v\n", err)
		}
		log.Printf("migrated [%d] %s", startIdx+i, ghRepo.FullName)
	}
}] %s", startIdx+i, ghRepo.FullName)
	}
}