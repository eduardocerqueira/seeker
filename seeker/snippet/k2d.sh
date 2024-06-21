#date: 2024-06-21T17:00:16Z
#url: https://api.github.com/gists/cbd4e48abc763a737b11164f69eabc54
#owner: https://api.github.com/users/rjchicago

# K2D

# <https://docs.k2d.io/quick-start-guide>

SECRET= "**********"

docker run --rm -d \
  --name k2d-k2d \
  --network bridge \
  --publish 6443:6443 \
  --env K2D_ADVERTISE_ADDR=$(ipconfig getifaddr en0) \
  --env K2D_SECRET= "**********"
  --label resource.k2d.io/namespace-name=k2d \
  --label workload.k2d.io/name=k2d \
  --volume /var/run/docker.sock:/var/run/docker.sock \
  --volume ~/.k2d:/var/lib/k2d \
  portainer/k2d:1.0.0

# get k2d kubeconfig
curl --insecure -H "Authorization: "**********"://$(ipconfig getifaddr en0):6443/k2d/kubeconfig" > ~/.kube/k2d.yml

# export KUBECONFIG
export KUBECONFIG=~/.kube/config$(for YAML in $(find ${HOME}/.kube -name '*.y*ml') ; do echo -n ":${YAML}"; done)

# usage
kubectl config use-context k2d
kubectl describe pod k2d -n k2d

# cleanup
docker stop k2d-k2d
rm -rf ~/.k2d
```
~/.k2d
```
