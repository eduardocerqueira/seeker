#date: 2022-09-23T16:58:28Z
#url: https://api.github.com/gists/7df601bb72039db33e2abda0bd6bda34
#owner: https://api.github.com/users/glennswest

oc delete secret htpasswd -n openshift-config
rm -r -f users
mkdir users
cd users
touch htpasswd
htpasswd -Bb htpasswd admin Admin1!
htpasswd -Bb htpasswd gwest password
oc --user= "**********"=htpasswd -n openshift-config
oc replace -f - <<API
apiVersion: config.openshift.io/v1
kind: OAuth
metadata:
  name: cluster
spec:
  identityProviders:
  - name: "**********"
    mappingMethod: claim
    type: HTPasswd
    htpasswd:
      fileData:
        name: htpasswd
API
oc adm groups new mylocaladmins
oc adm groups add-users mylocaladmins admin gwest
oc adm policy add-cluster-role-to-group cluster-admin mylocaladmins
to-group cluster-admin mylocaladmins
