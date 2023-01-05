#date: 2023-01-05T17:10:52Z
#url: https://api.github.com/gists/c23991984ac10bbe86eaee20d95bd245
#owner: https://api.github.com/users/adamf

#!/bin/bash
set -exu -o pipefail


# This script creates a highly-available VPN between a VPC in Google Cloud Platform 
# and a VPC in AWS.  This does all of the work (and a bit more) detailed in
# https://cloud.google.com/architecture/build-ha-vpn-connections-google-cloud-aws

export AWS_PAGER=""

# When this is moved to infrastructure-as-code, you should have different shared secrets per tunnel.
# It's worth noting that GCP-generated shared secrets can contain characters that are invalid
# in AWS shared secrets.
SHARED_SECRET= "**********"

# Remeber to change subnets & ASNs
PREFIX=staging
GCP_PROJECT=<your-GCP-project>
GCP_REGION=asia-southeast1
GCP_VPC_NAME=${PREFIX}-gcp-to-aws-vpc
GCP_SUBNET_NAME=${PREFIX}-gcp-to-aws-subnet

# 10.128.0.0/9 is the IP range used by VPCs in GCP with automatic assignment
# turned on.
GCP_IP_RANGE=10.128.0.0/9
GCP_HA_VPN_GATEWAY_NAME=${PREFIX}-gcp-to-aws-ha-vpn-gateway
GCP_ROUTER_NAME=${PREFIX}-gcp-to-aws-cloud-router

# Any private ASN in the range 64512-65534 or 4200000000-4294967294
GCP_ASN=65500
GCP_PEER_GATEWAY_NAME=${PREFIX}-gcp-to-aws-peer-gateway

AWS_REGION=ap-southeast-1
# Any private ASN in the range 64512-65534 or 4200000000-4294967294
AWS_ASN=65001
AWS_VPC=${PREFIX}-aws-to-gcp-vpc

# Pick a reasonable range here; don't overlap with the GCP range.
AWS_IP_RANGE=10.3.0.0/16

AWS_VPN_CONNECTION_NAME_1=${PREFIX}-aws-to-gcp-vpn-conn-1
AWS_VPN_CONNECTION_NAME_2=${PREFIX}-aws-to-gcp-vpn-conn-2
AWS_IGW_NAME=${PREFIX}-aws-to-gcp-igw

AWS_T1_CIDR=169.254.100.0/30
AWS_T1_IP=169.254.100.1
GCP_T1_IP=169.254.100.2
AWS_T2_CIDR=169.254.100.4/30
AWS_T2_IP=169.254.100.5
GCP_T2_IP=169.254.100.6
AWS_T3_CIDR=169.254.100.8/30
AWS_T3_IP=169.254.100.9
GCP_T3_IP=169.254.100.10
AWS_T4_CIDR=169.254.100.12/30
AWS_T4_IP=169.254.100.13
GCP_T4_IP=169.254.100.14

gcloud compute networks create ${GCP_VPC_NAME} \
    --project ${GCP_PROJECT} \
    --subnet-mode custom \
    --bgp-routing-mode global

gcloud compute networks subnets create ${GCP_SUBNET_NAME} \
    --network ${GCP_VPC_NAME} \
    --project ${GCP_PROJECT} \
		--region ${GCP_REGION} \
		--range ${GCP_IP_RANGE}

gcloud compute vpn-gateways create ${GCP_HA_VPN_GATEWAY_NAME} \
    --project ${GCP_PROJECT} \
    --network ${GCP_VPC_NAME} \
    --region ${GCP_REGION}

gcloud compute routers create ${GCP_ROUTER_NAME} \
    --project ${GCP_PROJECT} \
    --region ${GCP_REGION} \
    --network ${GCP_VPC_NAME} \
    --asn ${GCP_ASN} \
    --advertisement-mode custom \
    --set-advertisement-groups all_subnets


gcloud compute routers describe ${GCP_ROUTER_NAME} \
    --project ${GCP_PROJECT} \
    --region ${GCP_REGION}  \
    --format text

GCP_VPN_INTERFACE_0=$(gcloud compute vpn-gateways describe ${GCP_HA_VPN_GATEWAY_NAME} \
    --project ${GCP_PROJECT} \
    --region ${GCP_REGION} \
    --format text | grep ipAddress | grep "\[0\]" | cut -d' ' -f2)
GCP_VPN_INTERFACE_1=$(gcloud compute vpn-gateways describe ${GCP_HA_VPN_GATEWAY_NAME} \
    --project ${GCP_PROJECT} \
    --region ${GCP_REGION} \
    --format text | grep ipAddress | grep "\[1\]" | cut -d' ' -f2)

aws ec2 create-vpc \
    --region=${AWS_REGION} \
    --cidr-block ${AWS_IP_RANGE} \
    --tag-specification "ResourceType=vpc,Tags=[{Key=Name,Value=${AWS_VPC}}]"

AWS_VPC_ID=$(aws ec2 describe-vpcs --region=${AWS_REGION} --filters "Name=tag:Name,Values=${AWS_VPC}" --query "Vpcs[0].VpcId" --output text)
AWS_VPC_ROUTE_TABLE_ID=$(aws ec2 describe-route-tables --region=${AWS_REGION} --filters "Name=vpc-id,Values=${AWS_VPC_ID}" "Name=association.main,Values=true" --query "RouteTables[*].RouteTableId" --output text)

aws ec2 create-internet-gateway --region=${AWS_REGION} --tag-specifications "ResourceType=internet-gateway,Tags=[{Key=Name,Value=${AWS_IGW_NAME}}]"

aws ec2 create-customer-gateway --type ipsec.1 --public-ip ${GCP_VPN_INTERFACE_0} --bgp-asn ${GCP_ASN} --region=${AWS_REGION}
AWS_CG_1=$(aws ec2 describe-customer-gateways --region ${AWS_REGION} --filters "Name=bgp-asn,Values=65000" --filters "Name=state,Values=available" --filters "Name=ip-address,Values=${GCP_VPN_INTERFACE_0}" --query "CustomerGateways[*].CustomerGatewayId" --output text | cut -f1)

aws ec2 create-customer-gateway --type ipsec.1 --public-ip ${GCP_VPN_INTERFACE_1} --bgp-asn ${GCP_ASN} --region=${AWS_REGION}
AWS_CG_2=$(aws ec2 describe-customer-gateways --region ${AWS_REGION} --filters "Name=bgp-asn,Values=65000" --filters "Name=state,Values=available" --filters "Name=ip-address,Values=${GCP_VPN_INTERFACE_1}" --query "CustomerGateways[*].CustomerGatewayId" --output text | cut -f1)

aws ec2 create-vpn-gateway --type ipsec.1 --amazon-side-asn ${AWS_ASN} --region=${AWS_REGION}

AWS_VPN_GATEWAY_ID=$(aws ec2 describe-vpn-gateways --region=${AWS_REGION} --filters "Name=amazon-side-asn,Values=${AWS_ASN}" --filters "Name=state,Values=available" --query "VpnGateways[0].VpnGatewayId" --output text)
AWS_IGW_ID=$(aws ec2 describe-internet-gateways --region=${AWS_REGION} --filters "Name=tag:Name,Values=${AWS_IGW_NAME}"  --query "InternetGateways[0].InternetGatewayId" --output text)


aws ec2 attach-vpn-gateway --region=${AWS_REGION} --vpn-gateway-id ${AWS_VPN_GATEWAY_ID} --vpc-id ${AWS_VPC_ID}
aws ec2 attach-internet-gateway --region=${AWS_REGION} --vpc-id ${AWS_VPC_ID} --internet-gateway-id ${AWS_IGW_ID}


aws ec2 create-vpn-connection \
		--type ipsec.1 \
		--region ${AWS_REGION} \
   	--customer-gateway-id ${AWS_CG_1} \
		--vpn-gateway-id ${AWS_VPN_GATEWAY_ID} \
		--no-paginate \
    --tag-specification "ResourceType=vpn-connection,Tags=[{Key=Name,Value=${AWS_VPN_CONNECTION_NAME_1}}]" \
		--options TunnelOptions= "**********"=${AWS_T1_CIDR},PreSharedKey=${SHARED_SECRET}},{TunnelInsideCidr=${AWS_T2_CIDR},PreSharedKey=${SHARED_SECRET}}]"

aws ec2 create-vpn-connection \
		--type ipsec.1 \
		--region ${AWS_REGION} \
		--customer-gateway-id ${AWS_CG_2} \
		--vpn-gateway-id ${AWS_VPN_GATEWAY_ID} \
		--no-paginate \
    --tag-specification "ResourceType=vpn-connection,Tags=[{Key=Name,Value=${AWS_VPN_CONNECTION_NAME_2}}]" \
		--options TunnelOptions= "**********"=${AWS_T3_CIDR},PreSharedKey=${SHARED_SECRET}},{TunnelInsideCidr=${AWS_T4_CIDR},PreSharedKey=${SHARED_SECRET}}]"

AWS_GW_IP_1=$(aws ec2 describe-vpn-connections --region ${AWS_REGION} --query "VpnConnections[?State != \`deleted\`].Options.TunnelOptions[*].OutsideIpAddress" --filters "Name=tag:Name,Values=${AWS_VPN_CONNECTION_NAME_1}" --output text | cut -f1)
AWS_GW_IP_2=$(aws ec2 describe-vpn-connections --region ${AWS_REGION} --query "VpnConnections[?State != \`deleted\`].Options.TunnelOptions[*].OutsideIpAddress" --filters "Name=tag:Name,Values=${AWS_VPN_CONNECTION_NAME_1}" --output text | cut -f2)
CG_ID=$(aws ec2 describe-vpn-connections --region ${AWS_REGION} --query "VpnConnections[?State != \`deleted\`].CustomerGatewayId" --filters "Name=tag:Name,Values=${AWS_VPN_CONNECTION_NAME_1}" --output text)
GCP_IP_FOR_AWS_GW_IPS_12=$(aws ec2 describe-customer-gateways --region ${AWS_REGION} --customer-gateway-ids ${CG_ID} --query "CustomerGateways[0].IpAddress" --filters "Name=state,Values=available" --output text)

AWS_GW_IP_3=$(aws ec2 describe-vpn-connections --region ${AWS_REGION} --query "VpnConnections[?State != \`deleted\`].Options.TunnelOptions[*].OutsideIpAddress" --filters "Name=tag:Name,Values=${AWS_VPN_CONNECTION_NAME_2}" --output text | cut -f1)
AWS_GW_IP_4=$(aws ec2 describe-vpn-connections --region ${AWS_REGION} --query "VpnConnections[?State != \`deleted\`].Options.TunnelOptions[*].OutsideIpAddress" --filters "Name=tag:Name,Values=${AWS_VPN_CONNECTION_NAME_2}" --output text | cut -f2)
CG_ID=$(aws ec2 describe-vpn-connections --region ${AWS_REGION} --query "VpnConnections[?State != \`deleted\`].CustomerGatewayId" --filters "Name=tag:Name,Values=${AWS_VPN_CONNECTION_NAME_1}" --output text)
GCP_IP_FOR_AWS_GW_IPS_34=$(aws ec2 describe-customer-gateways --region ${AWS_REGION} --customer-gateway-ids ${CG_ID} --query "CustomerGateways[0].IpAddress" --filters "Name=state,Values=available" --output text)


gcloud compute external-vpn-gateways --project=${GCP_PROJECT} create  ${GCP_PEER_GATEWAY_NAME} --interfaces \
    0=${AWS_GW_IP_1},1=${AWS_GW_IP_2},2=${AWS_GW_IP_3},3=${AWS_GW_IP_4}

INTERFACE=0
if [[ "${GCP_VPN_INTERFACE_0}" == "${GCP_IP_FOR_AWS_GW_IPS_12}" ]]; then
    INTERFACE=0
else
    INTERFACE=1
fi

# For all of these tunnels, the external interface number (starting at 0)
# must match a few things: the interface number in the external-vpn-gateways command above
# The shared secret on AWS tunnel that matches the AWS_GW_IP from the external-vpn-gateways
# command above, and the GCP IP used by the AWS customer gateway. If these _all_ don't match
# you're not going to VPN today.
gcloud compute vpn-tunnels create tunnel-1 \
		--project ${GCP_PROJECT} \
    --peer-external-gateway ${GCP_PEER_GATEWAY_NAME} \
    --peer-external-gateway-interface 0 \
    --region ${GCP_REGION}  \
    --ike-version 2 \
    --shared-secret ${SHARED_SECRET} \
    --router ${GCP_ROUTER_NAME} \
    --vpn-gateway ${GCP_HA_VPN_GATEWAY_NAME} \
    --interface ${INTERFACE} 

gcloud compute vpn-tunnels create tunnel-2 \
		--project ${GCP_PROJECT} \
    --peer-external-gateway ${GCP_PEER_GATEWAY_NAME} \
    --peer-external-gateway-interface 1 \
    --region ${GCP_REGION}  \
    --ike-version 2 \
    --shared-secret ${SHARED_SECRET} \
    --router ${GCP_ROUTER_NAME} \
    --vpn-gateway ${GCP_HA_VPN_GATEWAY_NAME} \
    --interface  ${INTERFACE}

if [[ ${INTERFACE} -eq 0 ]]; then
    INTERFACE=1
else
    INTERFACE=0
fi
gcloud compute vpn-tunnels create tunnel-3 \
		--project ${GCP_PROJECT} \
    --peer-external-gateway ${GCP_PEER_GATEWAY_NAME} \
    --peer-external-gateway-interface 2 \
    --region ${GCP_REGION}  \
    --ike-version 2 \
    --shared-secret ${SHARED_SECRET} \
    --router ${GCP_ROUTER_NAME} \
    --vpn-gateway ${GCP_HA_VPN_GATEWAY_NAME} \
    --interface ${INTERFACE}

gcloud compute vpn-tunnels create tunnel-4 \
		--project ${GCP_PROJECT} \
    --peer-external-gateway ${GCP_PEER_GATEWAY_NAME} \
    --peer-external-gateway-interface 3 \
    --region ${GCP_REGION}  \
    --ike-version 2 \
    --shared-secret ${SHARED_SECRET} \
    --router ${GCP_ROUTER_NAME} \
    --vpn-gateway ${GCP_HA_VPN_GATEWAY_NAME} \
    --interface ${INTERFACE}

gcloud compute routers add-interface ${GCP_ROUTER_NAME} \
		--project ${GCP_PROJECT} \
    --interface-name int-1 \
    --vpn-tunnel tunnel-1 \
    --ip-address ${GCP_T1_IP} \
    --mask-length 30 \
    --region ${GCP_REGION}

gcloud compute routers add-interface ${GCP_ROUTER_NAME} \
		--project ${GCP_PROJECT} \
    --interface-name int-2 \
    --vpn-tunnel tunnel-2 \
    --ip-address ${GCP_T2_IP} \
    --mask-length 30 \
    --region ${GCP_REGION}

gcloud compute routers add-interface ${GCP_ROUTER_NAME} \
		--project ${GCP_PROJECT} \
    --interface-name int-3 \
    --vpn-tunnel tunnel-3 \
    --ip-address ${GCP_T3_IP} \
    --mask-length 30 \
    --region ${GCP_REGION}

gcloud compute routers add-interface ${GCP_ROUTER_NAME} \
		--project ${GCP_PROJECT} \
    --interface-name int-4 \
    --vpn-tunnel tunnel-4 \
    --ip-address ${GCP_T4_IP} \
    --mask-length 30 \
    --region ${GCP_REGION}

gcloud compute routers add-bgp-peer ${GCP_ROUTER_NAME} \
		--project ${GCP_PROJECT} \
    --peer-name aws-conn1-tunn1 \
    --peer-asn ${AWS_ASN} \
    --interface int-1 \
    --peer-ip-address ${AWS_T1_IP} \
    --region ${GCP_REGION}

gcloud compute routers add-bgp-peer ${GCP_ROUTER_NAME} \
		--project ${GCP_PROJECT} \
    --peer-name aws-conn1-tunn2 \
    --peer-asn ${AWS_ASN} \
    --interface int-2 \
    --peer-ip-address ${AWS_T2_IP} \
    --region ${GCP_REGION}


gcloud compute routers add-bgp-peer ${GCP_ROUTER_NAME} \
		--project ${GCP_PROJECT} \
    --peer-name aws-conn2-tunn1 \
    --peer-asn ${AWS_ASN} \
    --interface int-3 \
    --peer-ip-address ${AWS_T3_IP} \
    --region ${GCP_REGION}


gcloud compute routers add-bgp-peer ${GCP_ROUTER_NAME} \
		--project ${GCP_PROJECT} \
    --peer-name aws-conn2-tunn2 \
    --peer-asn ${AWS_ASN} \
    --interface int-4 \
    --peer-ip-address ${AWS_T4_IP} \
    --region ${GCP_REGION}

aws ec2 create-route --region ${AWS_REGION} --route-table-id ${AWS_VPC_ROUTE_TABLE_ID} --destination-cidr-block ${GCP_IP_RANGE} --gateway-id ${AWS_VPN_GATEWAY_ID}
teway-id ${AWS_VPN_GATEWAY_ID}
