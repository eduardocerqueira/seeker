#date: 2022-09-27T17:13:09Z
#url: https://api.github.com/gists/fb0b08af5b43c64450f5ef147bccbd06
#owner: https://api.github.com/users/vijaykushwaha15

#!/bin/bash
RANCHERENDPOINT=https://your_rancher_endpoint/v3
# The name of the cluster where the user needs to be added
CLUSTERNAME=your_cluster_name
# Username, password and realname of the user
USERNAME=username
PASSWORD= "**********"
REALNAME=myrealname
# Role of the user
GLOBALROLE=user
CLUSTERROLE=cluster-member
# Admin bearer token to create user
ADMINBEARERTOKEN=token- "**********" "**********" "**********" "**********" "**********": "**********"

# Create user and assign role
USERID=`curl -s -u $ADMINBEARERTOKEN $RANCHERENDPOINT/user -H 'content-type: "**********":false,"mustChangePassword":false,"type":"user","username":"'$USERNAME'","password":"'$PASSWORD'","name":"'$REALNAME'"}' --insecure | jq -r .id`
curl -s -u $ADMINBEARERTOKEN $RANCHERENDPOINT/globalrolebinding -H 'content-type: "**********":"globalRoleBinding","globalRoleId":"'$GLOBALROLE'","userId":"'$USERID'"}' --insecure

# Get clusterid from name
CLUSTERID= "**********"=$CLUSTERNAME --insecure | jq -r .data[].id`

# Add user as member to cluster
curl -s -u $ADMINBEARERTOKEN $RANCHERENDPOINT/clusterroletemplatebinding -H 'content-type: "**********":"clusterRoleTemplateBinding","clusterId":"'$CLUSTERID'","userPrincipalId":"local://'$USERID'","roleTemplateId":"'$CLUSTERROLE'"}' --insecure

# Login as user and get usertoken
LOGINRESPONSE=`curl -s $RANCHERENDPOINT/v3-public/localProviders/local?action=login -H 'content-type: "**********":"'$USERNAME'","password":"'$PASSWORD'"}' --insecure`
USERTOKEN= "**********"

# Generate and save kubeconfig
curl -s -u $USERTOKEN $RANCHERENDPOINT/clusters/$CLUSTERID?action=generateKubeconfig -X POST -H 'content-type: "**********"

# Set mustChangePassword to true for user to change password upon login
curl -s -u $ADMINBEARERTOKEN $RANCHERENDPOINT/users/$USERID -X PUT -H 'content-type: "**********":true}' --insecurer user to change password upon login
curl -s -u $ADMINBEARERTOKEN $RANCHERENDPOINT/users/$USERID -X PUT -H 'content-type: "**********":true}' --insecurey '{"mustChangePassword":true}' --insecure