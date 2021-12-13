#date: 2021-12-13T17:04:59Z
#url: https://api.github.com/gists/70f76c85e526d361dd11be4e476a540c
#owner: https://api.github.com/users/sergeysedoy97

# add helm repository: bitnami
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# install helm package: grafana
helm upgrade grafana bitnami/grafana \
--install \
--wait \
--namespace sergeysedoy97 \
--set admin.password=password \
--set plugins=vertamedia-clickhouse-datasource \
--set persistence.size=100Mi \
--set ingress.enabled=true \
--set-string ingress.annotations."dev\.okteto\.com/auto-ingress"=true \
--set-string ingress.annotations."dev\.okteto\.com/generate-host"=true

# install helm package: influxdb
helm upgrade influxdb bitnami/influxdb \
--install \
--wait \
--namespace sergeysedoy97 \
--set auth.admin.password=password \
--set persistence.size=100Mi \
--set ingress.enabled=true \
--set-string ingress.annotations."dev\.okteto\.com/auto-ingress"=true \
--set-string ingress.annotations."dev\.okteto\.com/generate-host"=true

# add helm repository: redash
helm repo add redash https://getredash.github.io/contrib-helm-chart/
helm repo update

# install helm package: redash
helm upgrade redash redash/redash \
--install \
--wait \
--namespace sergeysedoy97 \
--set redash.secretKey=secretkey \
--set redash.cookieSecret=cookiesecret \
--set postgresql.postgresqlDatabase=redash \
--set postgresql.postgresqlUsername=redash \
--set postgresql.postgresqlPassword=redash \
--set postgresql.persistence.size=100Mi \
--set redis.password=password \
--set redis.master.persistence.size=100Mi \
--set ingress.enabled=true \
--set ingress.hosts[0].paths[0]=/ \
--set-string ingress.annotations."dev\.okteto\.com/auto-ingress"=true \
--set-string ingress.annotations."dev\.okteto\.com/generate-host"=true \
--set postgresql.image.tag=10.7.0-r68 # hotfix

# ---- #
# TODO #
# ---- #

# add helm repository: vector
helm repo add vector https://helm.vector.dev
helm repo update

# add helm repository: grafana
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update