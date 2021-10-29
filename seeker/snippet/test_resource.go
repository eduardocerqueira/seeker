//date: 2021-10-29T16:44:40Z
//url: https://api.github.com/gists/bd4935b84722175c34f4adcd05975730
//owner: https://api.github.com/users/hugorut

package aws

import (
	"github.com/infracost/infracost/internal/resources/aws"
	"github.com/infracost/infracost/internal/schema"

	"strings"
)

func getTestResourceRegistryItem() *schema.RegistryItem {
	return &schema.RegistryItem{
		Name:  "test_resource",
		RFunc: newTestResource,
	}
}

func newTestResource(d *schema.ResourceData, u *schema.UsageData) *schema.Resource {
	t := &aws.TestResource{
		Address:       d.Address,
		Region:        d.Get("region").String(),
	}
	t.PopulateUsage(u)

	return t.BuildResource()
}
