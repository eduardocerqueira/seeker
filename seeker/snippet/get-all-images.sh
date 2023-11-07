#date: 2023-11-07T16:44:38Z
#url: https://api.github.com/gists/3d70c2243e674aa15d01d99bb74abc30
#owner: https://api.github.com/users/pjaudiomv

#!/usr/bin/env bash

normalize_image_name() {
    local name="$1"
    local registry
    if [[ "$name" == *\/* ]]; then
        registry=${name%%/*}
        if ! [[ "$registry" == *.* || "$registry" == localhost:[0-9]* ]]; then
            name="docker.io/${name}"
        fi
    else
        name="docker.io/library/${name}"
    fi
    echo "$name"
}

get_pods_images() {
    kubectl get pods --all-namespaces -o jsonpath="{.items[*].spec.containers[*].image}" | tr -s '[[:space:]]' '\n'
}

get_jobs_images() {
    kubectl get jobs --all-namespaces -o jsonpath="{.items[*].spec.template.spec.containers[*].image}" | tr -s '[[:space:]]' '\n'
}

get_cron_jobs_images() {
    kubectl get cronjobs --all-namespaces -o jsonpath="{.items[*].spec.jobTemplate.spec.template.spec.containers[*].image}" | tr -s '[[:space:]]' '\n'
}

images=()
while IFS= read -r image; do
    normalized_image=$(normalize_image_name "$image")
    images+=("$normalized_image")
done < <(get_pods_images && get_jobs_images && get_cron_jobs_images)

unique_images=($(echo "${images[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

# Json
# printf '%s\n' "${unique_images[@]}" | jq -R . | jq -s '{images: map({name: .})}'

# Yaml
echo "images:"
for img in "${unique_images[@]}"; do
    echo "  - name: $img"
done
