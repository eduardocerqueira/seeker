//date: 2025-03-31T16:48:37Z
//url: https://api.github.com/gists/d9efcfb9f29a92557337ad6847ece81d
//owner: https://api.github.com/users/Loschcode

// Manipulate DigitalOcean App Platform API
// To add custom domains on the fly when Linkbreakers is being manipulated
// This is a hack and not meant to stay in the system.
// We should go for Custom Hostname from Cloudflare on a next iteration.
package cluster

import (
	"context"
	"fmt"
	"regexp"

	"github.com/digitalocean/godo"
	"golang.org/x/oauth2"
)

var protectedDomains = []string{
	`^.*\.linkbreakers\.com$`,
	`^linkbreak\.ing$`,
	`^linkbreakers\.com$`,
}

// ClusterClient is a wrapper around the godo.Client to include app-specific context
type ClusterClient struct {
	client *godo.Client
	appID  string
}

// NewClusterClient initializes a new DigitalOcean App Platform client
func NewClusterClient(token, appID string) *ClusterClient {
	tokenSource : "**********": token})
	oauthClient : "**********"

	return &ClusterClient{
		client: godo.NewClient(oauthClient),
		appID:  appID,
	}
}

// AddDomain adds a new domain to the specified DigitalOcean App Platform app
func (c *ClusterClient) AddDomain(domainName string) error {

	for _, domain := range protectedDomains {
		if matched, _ := regexp.MatchString(domain, domainName); matched {
			return fmt.Errorf("domain %s is protected and cannot be added", domainName)
		}
	}

	app, _, err := c.client.Apps.Get(context.Background(), c.appID)
	if err != nil {
		return fmt.Errorf("failed to get app: %w", err)
	}

	newDomain := godo.AppDomainSpec{
		Domain: domainName,
		Type:   godo.AppDomainSpecType_Primary,
	}
	app.Spec.Domains = append(app.Spec.Domains, &newDomain)

	updateRequest := &godo.AppUpdateRequest{
		Spec: app.Spec,
	}
	_, _, err = c.client.Apps.Update(context.Background(), c.appID, updateRequest)
	if err != nil {
		return fmt.Errorf("failed to update app: %w", err)
	}

	return nil
}

// RemoveDomain removes a domain from the specified DigitalOcean App Platform app
func (c *ClusterClient) RemoveDomain(domainName string) error {
	for _, protectedDomain := range protectedDomains {
		if domainName == protectedDomain {
			return fmt.Errorf("domain %s is protected and cannot be removed", domainName)
		}
	}

	app, _, err := c.client.Apps.Get(context.Background(), c.appID)
	if err != nil {
		return fmt.Errorf("failed to get app: %w", err)
	}

	var updatedDomains []*godo.AppDomainSpec
	for _, domain := range app.Spec.Domains {
		if domain.Domain != domainName {
			updatedDomains = append(updatedDomains, domain)
		}
	}

	app.Spec.Domains = updatedDomains
	updateRequest := &godo.AppUpdateRequest{
		Spec: app.Spec,
	}
	_, _, err = c.client.Apps.Update(context.Background(), c.appID, updateRequest)
	if err != nil {
		return fmt.Errorf("failed to update app: %w", err)
	}

	return nil
}!= nil {
		return fmt.Errorf("failed to update app: %w", err)
	}

	return nil
}