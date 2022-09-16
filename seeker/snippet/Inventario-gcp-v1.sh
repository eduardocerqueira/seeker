#date: 2022-09-16T22:26:26Z
#url: https://api.github.com/gists/222d52e265f31802e70de4083fde20f6
#owner: https://api.github.com/users/desigua

 #!/bin/bash
echo Script para hacer que trabajamos
sleep 5s
gcloud compute project-info describe >./project-info.txt
gcloud compute snapshots list >./snapshots.txt
gcloud compute networks list >./network-list.txt
gcloud compute firewall-rules list >./firewall-rules-list.txt
gcloud container images list >./images-list.txt
gcloud compute addresses list >./addresses-list.txt
gcloud compute routes list >./routes.txt
gcloud compute forwarding-rules list >./forwarding-rules.txt
gcloud compute target-pools list >./target-pools-list.txt
gcloud compute health-checks list >./hchecks-list.txt
gcloud compute target-instances list >./target-instances-list.txt
gcloud compute target-http-proxies list >./target-http-proxies.txt
gcloud compute target-https-proxies list >./target-https-proxies.txt
gcloud compute url-maps list >./url-maps-list.txt
gcloud compute backend-buckets list >./backend-bucket-list.txt
gcloud compute instance-templates list >./instance-templates-list.txt
gcloud compute target-vpn-gateways list >./target-vpn-gateways.txt
gcloud compute vpn-tunnels list >./vpn-tunnels-list.txt
gcloud compute backend-buckets list >./backend-buckets.txt
gcloud compute routers list >./routers-list.txt
gcloud compute target-ssl-proxies list >./target-ssl-proxies-list.txt
gcloud compute ssl-certificates list >./ssl-cert-list.txt
gcloud compute networks subnets list >./networks-subnets.txt
gcloud compute target-tcp-proxies list >./target-tcp-proxies-list.txt
gcloud compute security-policies list >./security-policies-list.txt
#gcloud compute security-policies describe <name-rules> >./armor-list-1.txt
gcloud compute packet-mirrorings list >./packet-mirrorings-list.txt
gcloud endpoints services list >./endpoint-list.txt
gcloud compute interconnects list >./interconnects-list.txt
gcloud compute vpn-gateways list >./vpn-gateways-list.txt
gcloud compute images list >./images-list.txt
gcloud compute external-vpn-gateways list >./external-vpn.gateways-list.txt