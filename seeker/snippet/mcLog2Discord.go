//date: 2021-09-21T17:04:13Z
//url: https://api.github.com/gists/a35b17aa4829cfbc8fbc4f7e097923a9
//owner: https://api.github.com/users/mattzque

package main

import (
	"fmt"
	"github.com/hpcloud/tail"
	"github.com/spf13/cobra"
	"net/http"
	"net/url"
	"os"
	"regexp"
)

var VALID_MESSAGES = []string{
	`^\w+ joined the game`,
	`^\<\w+\> .*`, // normal chat messages
	`^\* \w+ .*`,  // /me chat messages
	`^\w+ left the game`,
	`^Can't keep up! Is the server overloaded\?`,
}

func parseLogMessage(line string) string {
	logline := regexp.MustCompile(`^\[([^\]]+)\] \[([^\]]+)\]: (.*)$`)
	parts := logline.FindStringSubmatch(line)
	if len(parts) == 4 {
		message := parts[3]
		for _, pattern := range VALID_MESSAGES {
			if regexp.MustCompile(pattern).MatchString(message) {
				return message
			}
		}
	}
	return ""
}

func sendDiscordMessage(webhookUrl string, message string) error {
	response, err := http.PostForm(webhookUrl, url.Values{"content": {message}})
	if err != nil {
		return err
	}
	response.Body.Close()
	return nil
}

func main() {
	var webhookUrl string
	var logfile string
	var cmd = &cobra.Command{
		Use:   "mcLogToDiscord",
		Short: "Read minecraft server log and post to Discord",
		Run: func(cmd *cobra.Command, args []string) {
			t, err := tail.TailFile(logfile, tail.Config{Follow: true, Location: &tail.SeekInfo{Offset: 0, Whence: 2}})
			if err != nil {
				fmt.Fprintf(os.Stderr, "error reading minecraft logfile: %v\n", err)
				return
			}
			for line := range t.Lines {
				message := parseLogMessage(line.Text)
				if message == "" {
					continue
				}
				err = sendDiscordMessage(webhookUrl, message)
				if err != nil {
					fmt.Fprintf(os.Stderr, "failed to post to discord webhook: %v\n", err)
				}
			}
		},
	}
	cmd.Flags().StringVarP(&webhookUrl, "url", "u", "", "Discord WebHook URL.")
	cmd.MarkFlagRequired("url")
	cmd.Flags().StringVarP(&logfile, "logfile", "l", "logs/latest.log", "Minecraft Server Logfile (minecraft.log)")
	if err := cmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}