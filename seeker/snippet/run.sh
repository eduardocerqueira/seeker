#date: 2022-10-21T17:07:24Z
#url: https://api.github.com/gists/1d467372363d772b62aef737dc99738a
#owner: https://api.github.com/users/idemery

# Create secret in devops namespace
kubectl -n devops create secret generic mssql --from-literal= "**********"="MyC0m9l&xP@ssw0rd"
# Deploy MS SQL Server
kubectl -n devops apply -f sql.yml.yml