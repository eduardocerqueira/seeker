#date: 2022-10-14T17:23:15Z
#url: https://api.github.com/gists/cbff5255dc7f67ad737703d8d72db834
#owner: https://api.github.com/users/nickyoung-hashicorp

# User Workflow
# Within GitHub
# 1. Create a new repo and provide a name

# Commit/Push to new repo
# 2. Clone existing repo with Terraform code or add your own initial .tf files
# 3. Open Terminal, `cd` to directory with .tf files
# 4. Run commands:
# ```sh
# git init
# git add *.tf
# git commit -m "first commit"
# git branch -M main
# git remote add origin <new_repo_url> [Example: https://github.com/nickyoung-hashicorp/dbaas-nullresource.git]
# git push -u origin main
# ```

# TFE Automation Script translatable to UI or CI/CD Pipeline

## Save Environment Variables
tfc_token= "**********"
address="app.terraform.io" # If using TFE, modify to the FQDN of TFE

########################
# 01) CREATE WORKSPACE #
########################

read -p "Provide the TFC/E organization name: " organization
read -p "Provide a workspace name: " workspace
read -p "Provide the repo name: " repo_name
vcs_repo="nickyoung-hashicorp\/$repo_name"
vcs_provider="GitHub.com/$organization"


echo "The TFC/E organization name is $organization"
sleep 1
echo "The workspace name is $workspace"
sleep 1
echo "The repo name is $repo_name"
sleep 1

# Set name of workspace in workspace.json (create a payload.json)
sed -e "s/placeholder/$workspace/" < workspace.template.json > workspace.json

# Create workspace
workspace_result=$(
  curl -Ss \
       --header "Authorization: "**********"
       --header "Content-Type: application/vnd.api+json" \
       --request POST \
	     --data @workspace.json \
       "https://${address}/api/v2/organizations/${organization}/workspaces"
)

echo "Workspace $workspace has been created" && echo "View the workspace at https://app.terraform.io/app/${organization}/workspaces/${workspace}/"

# 02) ASSIGN VARIABLES TO WORKSPACE
echo "View the missing variables for this workspace at https://app.terraform.io/app/${organization}/workspaces/${workspace}/variables"

read -p "Press [Enter] to update the workspace's variables"

# Add variables to workspace
while IFS=',' read -r key value category hcl sensitive
do
  sed -e "s/my-organization/$organization/" \
      -e "s/my-workspace/$workspace/" \
      -e "s/my-key/$key/" \
      -e "s/my-value/$value/" \
      -e "s/my-category/$category/" \
      -e "s/my-hcl/$hcl/" \
      -e "s/my-sensitive/$sensitive/" < variable.template.json  > variable.json

  echo "Adding variable $key in category $category "

  upload_variable_result=$(
    curl -Ss \
         --header "Authorization: "**********"
         --header "Content-Type: application/vnd.api+json" \
         --data @variable.json \
         "https://${address}/api/v2/vars?filter%5Borganization%5D%5Bname%5D=${organization}&filter%5Bworkspace%5D%5Bname%5D=${workspace}"
  )
done < variables.csv

echo "Variables have been assigned" && echo "View the variables for this workspace at https://app.terraform.io/app/${organization}/workspaces/${workspace}/variables"

# 03) ASSIGN VCS REPO TO WORKSPACE AND TRIGGER PLAN & APPLY
echo "View the missing VCS setup at https://app.terraform.io/app/${organization}/workspaces/${workspace}/settings/version-control"

read -p "Press [Enter] to configure the workspace with your VCP repo"

# Request the TF[C/E] VCS-Provider oauth-token
oauth_token= "**********"
  curl -Ss \
       --header "Authorization: "**********"
       --header "Content-Type: application/vnd.api+json" \
       --request GET \
       "https://${address}/api/v2/organizations/${organization}/oauth-clients" |\
 "**********"  "**********"  "**********"j "**********"q "**********"  "**********"- "**********"r "**********"  "**********"" "**********". "**********"d "**********"a "**********"t "**********"a "**********"[ "**********"] "**********"  "**********"| "**********"  "**********"s "**********"e "**********"l "**********"e "**********"c "**********"t "**********"  "**********"( "**********". "**********"a "**********"t "**********"t "**********"r "**********"i "**********"b "**********"u "**********"t "**********"e "**********"s "**********". "**********"n "**********"a "**********"m "**********"e "**********"  "**********"= "**********"= "**********"  "**********"\ "**********"" "**********"$ "**********"v "**********"c "**********"s "**********"_ "**********"p "**********"r "**********"o "**********"v "**********"i "**********"d "**********"e "**********"r "**********"\ "**********"" "**********") "**********"  "**********"| "**********"  "**********". "**********"r "**********"e "**********"l "**********"a "**********"t "**********"i "**********"o "**********"n "**********"s "**********"h "**********"i "**********"p "**********"s "**********". "**********"\ "**********"" "**********"o "**********"a "**********"u "**********"t "**********"h "**********"- "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"\ "**********"" "**********". "**********"d "**********"a "**********"t "**********"a "**********"[ "**********"] "**********". "**********"i "**********"d "**********"  "**********"" "**********"
)

# Setup VCS repo and additional parameters (auto-apply, queue run in workspace-vcs.json
sed -e "s/placeholder/$workspace/" \
    -e "s/vcs_repo/$vcs_repo/" \
    -e "s/oauth_token/$oauth_token/" < workspace-vcs.template.json  > workspace-vcs.json

read -p "Press [Enter] to run a Terraform Apply to provision infrastructure"

# Patch workspace and run the apply
apply_result=$(
  curl -Ss \
       --header "Authorization: "**********"
	     --header "Content-Type: application/vnd.api+json" \
	     --request PATCH \
	     --data @workspace-vcs.json \
	     "https://${address}/api/v2/organizations/${organization}/workspaces/${workspace}" | jq -r '.data.relationships."current-run".data.id')

echo "View the updated VCS setup at https://app.terraform.io/app/${organization}/workspaces/${workspace}/settings/version-control"

echo "Apply is automatically running..."

echo "Check the progress of your run at https://app.terraform.io/app/$organization/workspaces/$workspace/runs"

# 04) RETURN TO TERMINAL, MODIFY .TF FILES, AND COMMIT/PUSH
read -p "DEVELOPER WORKFLOW - MODIFY IAC VIA VERSION CONTROL SYSTEM.  Press [Enter] when VCS / Pull Request workflows are complete to proceed with destroying resources."

# 05) DESTROY RESOURCES, DELETE WORKSPACE
read -p "Before destroying, press [Enter] to retrieve the workspace ID"

# Get Workspace ID
workspace_id=$(curl \
  --header "Authorization: "**********"
  --header "Content-Type: application/vnd.api+json" \
  https://app.terraform.io/api/v2/organizations/${organization}/workspaces/${workspace} | jq -r '.data.id')

echo "The Workspace ID is: $workspace_id"

read -p "Press [Enter] to run a Terraform Destroy to delete the infrastructure"

# Set workspace ID from `destroy.template.json` and create `destroy.json`
sed -e "s/workspace_id/$workspace_id/" < destroy.template.json > destroy.json

echo "Terraform is destroying infrastructure..."

# Create a Destroy run
destroy_result=$(curl \
  --header "Authorization: "**********"
  --header "Content-Type: application/vnd.api+json" \
  --request POST \
  --data @destroy.json \
  https://app.terraform.io/api/v2/runs | jq -r '.data.id')

echo "Check the progress of your run at https://app.terraform.io/app/$organization/workspaces/$workspace/runs"

read -p "WAIT UNTIL THE INFRASTRUCTURE IS DESTROYED, then press [Enter] to proceed with deleting the workspace from Terraform Cloud / Enterprise"

echo "Attempting to delete the workspace"
delete_workspace_result=$(curl --header "Authorization: "**********": application/vnd.api+json" --request DELETE "https://${address}/api/v2/organizations/${organization}/workspaces/${workspace}")

echo "Cleaning up temporary JSON files"
rm destroy.json variable.json workspace-vcs.json workspace.json

# Get the response from the TFE server
# Note that successful deletion will give a null response.
# Only errors result in data.
echo "Response from TFE: ${delete_workspace_result} \n(A null response means the workspace is deleted)"