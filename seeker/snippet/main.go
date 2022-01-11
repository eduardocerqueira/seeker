//date: 2022-01-11T17:19:51Z
//url: https://api.github.com/gists/7800d5acad123c75c3e3262f3f4f4779
//owner: https://api.github.com/users/wahabmk

package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"syscall"
)

const (
	FillBomUID  = 1001
	ScanUID     = 1002
	SynopsysGID = 101
)

func command(ctx context.Context, cred *syscall.Credential, command string, args []string, params map[string]string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, command, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Credential: cred}
	cmd.Env = os.Environ()
	for k, v := range params {
		env := fmt.Sprintf("%s=%s", strings.ToUpper(k), v)
		cmd.Env = append(cmd.Env, env)
	}
	return cmd
}

func main() {
	ctx := context.Background()
	checkUserAndGroup := &syscall.Credential{Uid: FillBomUID, Gid: SynopsysGID, NoSetGroups: true}
	params := map[string]string{
		"PYTHONHASHSEED":     "0",
		"APPCHECK_NO_SYSLOG": "1",
		"PGSSLKEY":           "/tmp/postgres-client.1690483872.key",
		"POSTGRES_DBNAME":    "fuzzomatic",
		"POSTGRES_HOST":      "",
		"POSTGRES_PORT":      "",
		"POSTGRES_USER":      "",
		"POSTGRES_PASSWORD":  "",
	}

	cmd := command(ctx, checkUserAndGroup,
		"python3", []string{"/check/manage.py", "fillbom"}, params,
	)
	result, err := checkBOM(cmd, "/check/scan_results/_sha256:7b1a6ab2e44dbac178598dabe7cff59bd67233dba0b27e4fbd1f9d4b3c877a54.json")
	if err != nil {
		fmt.Printf("Error for /check/manage.py fillbom: %s\n", err)
	} else {
		fmt.Println(result)
	}

	fmt.Println("============================================================")

	cmd = command(ctx, nil, "cat", nil, nil)
	result, err = checkBOM(cmd, "")
	if err != nil {
		fmt.Printf("Error for cat: %s\n", err)
	} else {
		fmt.Printf("Output from cat: %s\n", result)
	}
}

func checkBOM(cmd *exec.Cmd, bomFile string) (string, error) {
	var err error
	input := []byte("hello world")
	if bomFile != "" {
		input, err = os.ReadFile(bomFile)
		if err != nil {
			return "", fmt.Errorf("Could not read file: %s", err)
		}
	}

	// ctx := context.Background()
	// args := []string{"/check/manage.py", "fillbom"}
	// args := []string{"hello world"}
	// cmd := command(ctx, checkUserAndGroup, "python3", args, params)
	// cmd := command(ctx, nil, cmnd, args, params)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return "", fmt.Errorf("Error create stdin pipe: %w", err)
	}

	var writeErr error
	var wg sync.WaitGroup
	go func() {
		wg.Add(1)
		defer wg.Done()
		defer stdin.Close()
		_, writeErr = io.WriteString(stdin, string(input))
	}()

	wg.Wait()
	checkBytes, err := cmd.Output()

	if err != nil {
		return "", fmt.Errorf("Error reading check data for layer, error: %w", err)
	}
	if writeErr != nil {
		return "", fmt.Errorf("Error writing to stdin pipe: %w", writeErr)
	}

	return string(checkBytes), nil
}
