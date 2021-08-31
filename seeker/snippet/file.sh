#date: 2021-08-31T01:10:01Z
#url: https://api.github.com/gists/2ec77c05c53aa59135833e7ff0743f5c
#owner: https://api.github.com/users/gcastellan0s

[OSEv3:children]
masters
nodes
etcd
lb

[OSEv3:vars]
ansible_ssh_user=root
openshift_deployment_type=origin
openshift_release=v3.6
openshift_image_tag=v3.6.1
openshift_pkg_version=-3.6.1
openshift_master_identity_providers=[{'name': 'htpasswd_auth', 'login': 'true', 'challenge': 'true', 'kind': 'HTPasswdPasswordIdentityProvider', 'filename': '/etc/origin/master/htpasswd'}]
openshift_master_htpasswd_users={'admin': '$apr1$gfaL16Jf$c.5LAvg3xNDVQTkk6HpGB1'}
etcd_version="3.1.9"
openshift_disable_check=disk_availability,docker_storage
docker_selinux_enabled=false
openshift_docker_options=" --log-driver=journald --storage-driver=overlay --registry-mirror=http://4a0fee72.m.daocloud.io "
openshift_hosted_router_selector='region=infra,router=true'
openshift_master_default_subdomain=app.imss.gob.mx

[masters]
master.imss.gob.mx

[etcd]
master.imss.gob.mx

[lb]
lb.imss.gob.mx

[nodes]
master.imss.gob.mx openshift_schedulable=true openshift_node_labels="{'region': 'infra', 'router': 'true'}"
node01.imss.gob.mx openshift_schedulable=true openshift_node_labels="{'region': 'infra', 'router': 'true'}"
node02.imss.gob.mx openshift_schedulable=true openshift_node_labels="{'region': 'infra', 'router': 'true'}"