//date: 2024-04-01T17:08:19Z
//url: https://api.github.com/gists/a61130c57a3d4558c46ad7d748ceb970
//owner: https://api.github.com/users/hardeepnarang10

package main

import (
	"context"
	"log"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/sso"
	"github.com/aws/aws-sdk-go-v2/service/ssooidc"
	"github.com/pkg/browser"
)

// This gist shows how to log into AWS SSO only using AWS SDK for Go. 
// It will launch a browser as if you had typed `aws sso login` on the command line.
// If successful will print the list of accounts you have access to.

// Change the startURL and region below to your AWS SSO start url and default
// region accordingly.
func main() {
	var (
		startURL string = "https://CHANGEME.awsapps.com/start"
		region          = "eu-west-2"
	)

	cfg, err := config.LoadDefaultConfig(context.TODO(), config.WithDefaultRegion(region))
	if err != nil {
		log.Fatalf("%v", err)
	}
	// create SSO oidcClient client to trigger login flow
	oidcClient := ssooidc.NewFromConfig(cfg)

	// register your client which is triggering the login flow
	register, err := oidcClient.RegisterClient(context.TODO(), &ssooidc.RegisterClientInput{
		ClientName: aws.String("sso-cli-client"),
		ClientType: aws.String("public"),
	})

	if err != nil {
		log.Fatal(err)
	}

	// authorize your device using the client registration response
	deviceAuth, err := oidcClient.StartDeviceAuthorization(context.TODO(), &ssooidc.StartDeviceAuthorizationInput{
		ClientId:     register.ClientId,
		ClientSecret: "**********"
		StartUrl:     aws.String(startURL),
	})
	if err != nil {
		log.Fatal(err)
	}

	url := aws.ToString(deviceAuth.VerificationUriComplete)
	log.Printf("If your browser is not opened automatically, please open link:\n%v\n", url)
	err = browser.OpenURL(url)
	if err != nil {
		log.Fatal(err)
	}

	var token *ssooidc.CreateTokenOutput
	approved := false

	// poll the client until it has finished authorization.
	for !approved {
		t, err : "**********"
			ClientId:     register.ClientId,
			ClientSecret: "**********"
			DeviceCode:   deviceAuth.DeviceCode,
			GrantType:    aws.String("urn:ietf:params:oauth:grant-type:device_code"),
		})
		if err != nil {
			isPending := strings.Contains(err.Error(), "AuthorizationPendingException:")
			if isPending {
				log.Println("Authorization pending...")
				time.Sleep(time.Duration(deviceAuth.Interval) * time.Second)
				continue
			}
		}
		approved = true
		token = "**********"
	}

	ssoClient := sso.NewFromConfig(cfg)

	log.Println("Fetching list of accounts for this user")
	accountPaginator := sso.NewListAccountsPaginator(ssoClient, &sso.ListAccountsInput{
		AccessToken: "**********"
	})

	for accountPaginator.HasMorePages() {
		x, err := accountPaginator.NextPage(context.TODO())
		if err != nil {
			log.Fatal(err)
		}
		for _, y := range x.AccountList {
			log.Println("-------------------------------------------------------")
			log.Printf("Account ID: %v Name: %v Email: %v\n", aws.ToString(y.AccountId), aws.ToString(y.AccountName), aws.ToString(y.EmailAddress))
		}
	}
}
 aws.ToString(y.AccountName), aws.ToString(y.EmailAddress))
		}
	}
}
