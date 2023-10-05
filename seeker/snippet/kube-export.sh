#date: 2023-10-05T16:57:02Z
#url: https://api.github.com/gists/7e8dacc2e4eb9c536e20599bc9366ca2
#owner: https://api.github.com/users/dirtboll

# Example exporting k8s lens metrics
OUTPUT_DIR=lens
KUBE_LABEL=app.kubernetes.io/managed-by=Lens
KUBE_NAMESPACE=

if ! command -v yq &> /dev/null; then
    echo "yq is requried, install here https://mikefarah.gitbook.io/yq/#install."
    exit 1
fi

mkdir -p $OUTPUT_DIR
kube_namespace=${KUBE_NAMESPACE:+-n $KUBE_NAMESPACE}
resource_kinds=$(kubectl api-resources ${KUBE_NAMESPACE:+--namespaced} --verbs=list -o name)
for reskind in ${resource_kinds[@]}; do
    IFS=$'\n'
    resources=($(
        kubectl get --show-kind --ignore-not-found ${kube_namespace:---all-namespaces} ${KUBE_LABEL:+-l "$KUBE_LABEL"} -o yaml $reskind |
            yq -P -o=j -I=0 '.items[] 
                | select( .metadata.annotations."kubectl.kubernetes.io/last-applied-configuration" != null ) 
                | .metadata.annotations."kubectl.kubernetes.io/last-applied-configuration" 
                | fromjson'
    ))
    for resource in ${resources[@]}; do
        kind=$(yq '.kind' <<< "$resource")
        name=$(yq '.metadata.name' <<< "$resource")
        dir="$OUTPUT_DIR/$(echo $kind | tr '[:upper:]' '[:lower:]')"
        mkdir -p $dir
        yq -Poy '.' <<< "$(printf "%s" "$resource")" > "$dir/$(echo $name | tr '[:upper:]' '[:lower:]').yml"
        echo "${kind}/${name}"
    done
done