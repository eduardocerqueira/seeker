//date: 2022-12-06T16:52:25Z
//url: https://api.github.com/gists/faff50c555340329a0b1eeed94145f24
//owner: https://api.github.com/users/iainlane

// wait-for-github-pr
// Copyright (C) 2022, Grafana Labs

// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU Affero General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version.

// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
// details.

// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

package main

import (
	"context"
	"crypto/rsa"
	"sync"
	"time"

	"github.com/beatlabs/github-auth/app"
	"github.com/beatlabs/github-auth/key"
	"github.com/shurcooL/githubv4"
	"github.com/urfave/cli/v2"

	log "github.com/sirupsen/logrus"

	"fmt"
	"os"
)

type authInfo struct {
	installationID string
	appID          string
	privateKey     *rsa.PrivateKey
}

func authenticate(ctx context.Context, auth authInfo) (*githubv4.Client, error) {
	// Create an App Config using the App ID and the private key
	app, err := app.NewConfig(auth.appID, auth.privateKey)
	if err != nil {
		return nil, err
	}

	// The client can be used to send authenticated requests
	install, err := app.InstallationConfig(auth.installationID)
	if err != nil {
		return nil, err
	}

	client := install.Client(ctx)
	githubClient := githubv4.NewClient(client)

	return githubClient, nil
}

func isPRClosed(ctx context.Context, githubClient *githubv4.Client, owner string, repo string, prNumber int) (bool, error) {
	var q struct {
		Repository struct {
			PullRequest struct {
				Closed bool
				Merged bool
			} `graphql:"pullRequest(number: $number)"`
		} `graphql:"repository(owner: $owner, name: $name)"`
	}

	variables := map[string]interface{}{
		"number": githubv4.Int(prNumber),
		"owner":  githubv4.String(owner),
		"name":   githubv4.String(repo),
	}

	err := githubClient.Query(ctx, &q, variables)
	if err != nil {
		return false, err
	}

	return q.Repository.PullRequest.Closed, nil
}

func main() {
	app := &cli.App{
		Name:  "wait-for-github-pr",
		Usage: "Wait for a GitHub PR to be merged",
		Flags: []cli.Flag{
			&cli.BoolFlag{
				Name:    "debug",
				Aliases: []string{"d"},
				Usage:   "Enable debug logging",
			},
			&cli.StringFlag{
				Name:    "github-app-private-key-file",
				Aliases: []string{"p"},
				Usage:   "Path to the GitHub App private key",
			},
			&cli.StringFlag{
				Name:    "github-app-private-key",
				Usage:   "Contents of the GitHub App private key",
				EnvVars: []string{"GITHUB_APP_PRIVATE_KEY"},
			},
			&cli.StringFlag{
				Name:     "github-app-id",
				Usage:    "GitHub App ID",
				EnvVars:  []string{"GITHUB_APP_ID"},
				Required: true,
			},
			&cli.StringFlag{
				Name:     "github-app-installation-id",
				Usage:    "GitHub App installation ID",
				EnvVars:  []string{"GITHUB_APP_INSTALLATION_ID"},
				Required: true,
			},
		},
		Action: func(c *cli.Context) error {
			var err error
			var privKey *rsa.PrivateKey

			formatter := &log.TextFormatter{
				FullTimestamp:   true,
				TimestampFormat: "2006-01-02 15:04:05",
			}
			log.SetFormatter(formatter)

			if c.Bool("debug") {
				log.SetLevel(log.DebugLevel)
				log.Debug("Debug logging enabled")
			}

			file := c.String("github-app-private-key-file")
			if file != "" {
				if privKey, err = key.FromFile(file); err != nil {
					return err
				}
				fmt.Println("file", file)
			}

			privateKey := c.String("github-app-private-key")
			if privateKey != "" {
				if privKey, err = key.Parse([]byte(privateKey)); err != nil {
					return err
				}
			}

			appId := c.String("github-app-id")
			installationID := c.String("github-app-installation-id")

			authInfo := authInfo{
				installationID: installationID,
				appID:          appId,
				privateKey:     privKey,
			}

			ctx := context.Background()
			githubClient, err := authenticate(ctx, authInfo)
			if err != nil {
				return err
			}

			wg := sync.WaitGroup{}
			ticker := time.NewTicker(1 * time.Minute)

			wg.Add(1)
			go func() {
				defer wg.Done()
				for range ticker.C {
					closed, err := isPRClosed(ctx, githubClient, "iainlane", "prometheus-launchpad-exporter", 46)
					if err != nil {
						log.Error(err)
						return
					}
					log.Println("closed", closed)

					if closed {
						return
					}
				}
			}()

			wg.Wait()

			return nil
		},
	}

	if err := app.Run(os.Args); err != nil {
		log.Fatal(err)
	}
}
