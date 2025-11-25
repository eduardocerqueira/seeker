#date: 2025-11-25T17:12:57Z
#url: https://api.github.com/gists/b38e6e1f9710da9fb417e277b755d6a3
#owner: https://api.github.com/users/quiiver

#!/bin/bash

set -eo pipefail

functions="webservices sandbox dataservices"
# functions="sandbox"
tenants="${1}"

gitroot=$(git rev-parse --show-toplevel)
argo_dir="$gitroot/argo/k8s/argocd-bootstrap"
pwd=$(pwd)
# tmpdir=$(mktemp -d)
tmpdir="$gitroot/tmpdir"
mkdir -p $tmpdir

diff_dir="$pwd/appset-diffs"
mkdir -p $diff_dir

# helpers for diff readability 
header='(.apiVersion = "argoproj.io/v1alpha1") | (.kind = "Application")'
filter='del(.metadata.annotations["notified.notifications.argoproj.io"]) |
  del(.metadata.managedFields) |
  del(.metadata.namespace) |
  del(.metadata.creationTimestamp) |
  del(.metadata.resourceVersion) |
  del(.metadata.uid) |
  del(.metadata.generation) |
  del(.metadata.ownerReferences) |
  del(.status)
'

cd ${tmpdir}
for func in $functions; do
  echo "Processing function: $func"
  namespace="argocd-$func"
  func_dir="$tmpdir/$func"
  if [[ -d "$func_dir" ]]; then
    rm -rf "$func_dir"
  fi
  mkdir -p "$func_dir"
  cd "$func_dir"
  echo "Setting namespace to $namespace"
  kubectl config set-context --current --namespace="$namespace"
  echo "Rendering appsets for function: $func"
  helm template $namespace $argo_dir -s "templates/tenants/tenant-applicationsets.yaml" -f "$argo_dir/values.yaml" -f "$argo_dir/$func.values.yaml" > "$func_dir/$func-appsets.yaml"
  while read -r index; do
    appset_file=$(mktemp)
    yq 'select(di == '$index')' $func_dir/$func-appsets.yaml > "$appset_file"
    appset_name=$(yq -r '.metadata.name' $appset_file)
    
    if [[ -n "$tenants" && "$appset_name" != *"$tenants"* ]]; then
      echo " -- Skipping appset: $appset_name"
      continue
    fi

    incoming_file="$func_dir/$appset_name-incoming.yaml"
    current_file="$func_dir/$appset_name-current.yaml"

    # select appset out of the big document
    echo " -- Found appset: $appset_name"
    argocd appset generate -o yaml $appset_file > "$incoming_file"
    if [[ $(yq '. | tag == "!!seq"' "$incoming_file") == "true" ]]; then
      echo " ---- Appset $appset_name generated multiple apps. splitting into docs."
      yq --inplace '.[] | split_doc' "$incoming_file"
    fi
    current_app_count=0
    touch "$current_file"
    while read -r app_name; do
      # add separator and header to argo output
      if [[ $current_app_count -gt 0 ]]; then
        echo "---" >> "$current_file"
      fi
      cat <(yq -n "$header") >> "$current_file"
      current_app_count=$((current_app_count + 1))

      echo " -- Fetching current app from ArgoCD: $app_name"
      argocd app get "$app_name" -o yaml \
          | yq "$filter" >> "$current_file"
    done < <(yq -r --no-doc '.metadata.name' "$incoming_file")
    yamlfmt "$current_file"
    yamlfmt "$incoming_file"

    diff_path="$diff_dir/${func}-${appset_name}.diff"
    echo " -- Writing diff to: $diff_path"
    diff -u "$current_file" "$incoming_file" > "$diff_path" || true
  done < <(yq -r 'di' $func_dir/$func-appsets.yaml)
  cd -
done
cd -

echo "Diffs written to: $diff_dir"
# rm -rf "$tmpdir"
