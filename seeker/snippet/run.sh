#date: 2025-11-04T17:08:01Z
#url: https://api.github.com/gists/046a074769f64075ee88fb898ba9db12
#owner: https://api.github.com/users/johuellm

# run.sh
# Cluster Config that exposes two NodePorts (30000 and 30001) to host machine
# Can easily be extended
# Prerequisites: docker for windows with WSL backend, kubectl, kind are installed (e.g. WSL2 Ubuntu-22.04)

# 1. Create kind Cluster
kind create cluster --config=cluster-config.yaml

# 2. Install kubernetes dashboard (on NodePort 30001)
helm install kubernetes-dashboard kubernetes-dashboard/kubernetes-dashboard -f dashboard.values.yaml

# 3. Create admin user
kubectl apply -f dashboard.user.yaml

# 4. Create login token for dashboard (manually copy output)
kubectl create token admin-user

# 5. Access dashboard
curl https://localhost:30001 --insecure    # (or use browser)

# 6. Apply whatever deployment you like. Configure so that NodePort 30000 is exposed
# 6a. For example nginx
kubectl apply -f nginx.yaml

# 6b. Or a custom deployment from open-webui (not included in this gist)
# kubectl create namespace open-webui-local
# kubectl apply -f complete-deployment.build.yaml
