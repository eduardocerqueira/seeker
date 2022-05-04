//date: 2022-05-04T16:58:50Z
//url: https://api.github.com/gists/7e3c52292eef0310ef34a398439dccb5
//owner: https://api.github.com/users/salrashid123

package main

/*

Using GCE VM:

assume adminapi@project.iam.gserviceaccount.com

1. has its client_id authorized for scopes
   https://www.googleapis.com/auth/admin.directory.user.readonly
   https://www.googleapis.com/auth/admin.directory.group.readonly
   https://www.googleapis.com/auth/cloud-platform


2. is authorized for read only operations for users and groups
 https://workspaceupdates.googleblog.com/2020/08/use-service-accounts-google-groups-without-domain-wide-delegation.html
 (in admin console, goto "Admin -> Admin roles" create custom role with "user read" and "group read"
  assign the svc account to that role


---

gcloud compute instances create dwdvmci \
  --service-account=adminapi@project.iam.gserviceaccount.com \
  --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/admin.directory.group.readonly  


in vm, install go,


$ curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/scopes
https://www.googleapis.com/auth/admin.directory.group.readonly
https://www.googleapis.com/auth/admin.directory.user.readonly

$ curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email
adminapi@project.iam.gserviceaccount.com

export TOKEN=`curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token | jq -r '.access_token'`

curl -H "Authorization: Bearer $TOKEN" https://admin.googleapis.com/admin/directory/v1/groups?domain=yourdomain.com


https://cloud.google.com/sdk/gcloud/reference/identity/groups/memberships/search-transitive-groups

*/

import (
	"fmt"
	"log"

	"context"

	"golang.org/x/oauth2/google"
	"google.golang.org/api/cloudidentity/v1"
	"google.golang.org/api/option"
)

func main() {

	ctx := context.Background()
	// serviceAccountFile := "/home/path/to/google_apps_svc_dwd.json"
	// serviceAccountJSON, err := ioutil.ReadFile(serviceAccountFile)
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// config, err := google.JWTConfigFromJSON(serviceAccountJSON, cloudidentity.CloudPlatformScope, cloudidentity.CloudIdentityGroupsReadonlyScope)
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// ts := config.TokenSource(ctx)

	ts, err := google.DefaultTokenSource(ctx)
	if err != nil {
		log.Fatal(err)
	}

	cisvc, err := cloudidentity.NewService(ctx, option.WithTokenSource(ts))
	if err != nil {
		log.Fatal(err)
	}

	g, err := cisvc.Groups.Memberships.SearchTransitiveGroups("groups/-").Query("member_key_id=='alice@domain.com' && 'cloudidentity.googleapis.com/groups.discussion_forum' in labels").Do()
	if err != nil {
		log.Fatal(err)
	}

	if len(g.Memberships) == 0 {
		fmt.Print("No groups found.\n")
	} else {
		fmt.Print("Groups:\n")
		for _, m := range g.Memberships {
			fmt.Printf("%s (%s)\n", m.GroupKey.Id, m.DisplayName)
		}
	}

}
