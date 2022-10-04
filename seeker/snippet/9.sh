#date: 2022-10-04T17:03:12Z
#url: https://api.github.com/gists/132573544bb9528b16a40d594cc409fd
#owner: https://api.github.com/users/jeromedecoster

# used to pull image from private ECR by argocd-image-updater
TOKEN= "**********"
kubectl create secret generic aws-ecr-creds \
    --from-literal=creds=AWS: "**********"
    --dry-run=client \
    --namespace argocd \
    --output yaml \
    | kubectl apply --filename -    | kubectl apply --filename -