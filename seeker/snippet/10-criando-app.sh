#date: 2022-07-22T17:11:24Z
#url: https://api.github.com/gists/0fe9c7c7e0a35d81fa71dd0a46774eea
#owner: https://api.github.com/users/AlexGalhardo

kubectl apply -f ./9-deploy.yaml
kubectl get pods
kubectl logs deploy/api
kubectl port-forward service/api 3000:80