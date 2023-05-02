#date: 2023-05-02T16:42:16Z
#url: https://api.github.com/gists/18a63d8238a9b03aaff264d449803baa
#owner: https://api.github.com/users/safoorsafdar

#!/usr/bin/env bash

#####################################################################
# REFERENCES
# - https://cloud.google.com/architecture/build-ha-vpn-connections-google-cloud-aws
# - https://cloud.google.com/vpc/docs/private-service-connect
#####################################################################

export PROJECT_ID=$(gcloud config get-value project)
export PROJECT_USER=$(gcloud config get-value core/account) # set current user
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
export IDNS=${PROJECT_ID}.svc.id.goog # workload identity domain

export GCP_REGION="us-west1" # CHANGEME (OPT)
export GCP_ZONE="us-west1-b" # CHANGEME (OPT)

# enable apis
gcloud services enable compute.googleapis.com \
    servicenetworking.googleapis.com
    
# configure gcloud sdk
gcloud config set compute/region $GCP_REGION
gcloud config set compute/zone $GCP_ZONE


############################################################################
# GCP NETWORK + VPN GATEWAY (GCP: 10.1.0.0/16) (us-west1: Oregon)
############################################################################

export NETWORK_NAME="gc-vpc"
export SUBNET_NAME="subnet-10-1-1-0"
export VPN_GATEWAY_NAME="vpn-gw-1"
export ROUTER_NAME="router-1"
export ASN_GCP="65510"
export GCP_RANGE="10.1.1.0/24"

# create custom vpc network
gcloud compute networks create $NETWORK_NAME \
    --subnet-mode custom \
    --bgp-routing-mode global

# create a test subnet
gcloud compute networks subnets create $SUBNET_NAME  \
    --network $NETWORK_NAME \
    --region $GCP_REGION \
    --range $GCP_RANGE

# create VPN gateway
gcloud compute vpn-gateways create $VPN_GATEWAY_NAME \
    --network $NETWORK_NAME \
    --region $GCP_REGION

# create cloud router
gcloud compute routers create $ROUTER_NAME \
    --region $GCP_REGION \
    --network $NETWORK_NAME \
    --asn $ASN_GCP \
    --advertisement-mode custom \
    --set-advertisement-groups all_subnets

# get public IP addresses from above (for use with AWS gateway setup)
export INTERFACE_0_IP_ADDRESS=$(gcloud compute vpn-gateways describe $VPN_GATEWAY_NAME --format="value(vpnInterfaces[0].ipAddress)")
export INTERFACE_1_IP_ADDRESS=$(gcloud compute vpn-gateways describe $VPN_GATEWAY_NAME --format="value(vpnInterfaces[1].ipAddress)")


############################################################################
# AWS NETWORK + VPN (AWS: 10.0.0.0/16) (us-west-1: N. California)
# - https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html 
# - https://docs.aws.amazon.com/vpc/latest/userguide/vpc-subnets-commands-example.html
############################################################################

export ASN_AWS="65520"
export AWS_RANGE="10.0.0.0/16"
export AWS_SUBNET_RANGE="10.0.0.0/24"
export TRANSIT_GATEWAY_DESCRIPTION="gateway-to-gcp"
export SHARED_SECRET= "**********"
export AWS_T1_IP="169.254.200" # cidr add 0./30
export AWS_T2_IP="169.254.201" # bgp IP add .2 ; peer IP add .1
export AWS_T3_IP="169.254.202"
export AWS_T4_IP="169.254.203"

# create custom vpc network
export AWS_VPC_ID=$(aws ec2 create-vpc --cidr-block $AWS_RANGE --query Vpc.VpcId --output text)

# create private route table
export PRIVATE_ROUTE_TABLE_ID=$(aws ec2 create-route-table \
    --vpc-id $AWS_VPC_ID \
    --query RouteTable.RouteTableId --output text)

# create a test subnet
export AWS_PRIVATE_SUBNET_ID=$(aws ec2 create-subnet --vpc-id $AWS_VPC_ID --cidr-block $AWS_SUBNET_RANGE \
    --query Subnet.SubnetId --output text)

# associate subnet to route table
aws ec2 associate-route-table \
    --route-table-id $PRIVATE_ROUTE_TABLE_ID \
    --subnet-id $AWS_PRIVATE_SUBNET_ID

# verify subnet created
aws ec2 describe-subnets --filters "Name=vpc-id,Values=$AWS_VPC_ID" \
    --query "Subnets[*].{ID:SubnetId,CIDR:CidrBlock}"

# create customer gateways
export AWS_CUSTOMER_GATEWAY_1=$(aws ec2 create-customer-gateway \
    --type ipsec.1 \
    --public-ip $INTERFACE_0_IP_ADDRESS \
    --bgp-asn $ASN_GCP \
    --query CustomerGateway.CustomerGatewayId --output text)

export AWS_CUSTOMER_GATEWAY_2=$(aws ec2 create-customer-gateway \
    --type ipsec.1 \
    --public-ip $INTERFACE_1_IP_ADDRESS \
    --bgp-asn $ASN_GCP \
    --query CustomerGateway.CustomerGatewayId --output text)

# create transit gateway
# (TIP: virtual private gateways cheaper option if don't need transit)
export AWS_TRANSIT_GATEWAY_ID=$(aws ec2 create-transit-gateway --description $TRANSIT_GATEWAY_DESCRIPTION \
    --options=AmazonSideAsn=$ASN_AWS,AutoAcceptSharedAttachments=enable,DefaultRouteTableAssociation=enable,DefaultRouteTablePropagation=enable,VpnEcmpSupport=enable,DnsSupport=enable \
    --query TransitGateway.TransitGatewayId --output text)

# attach transit gateway
aws ec2 create-transit-gateway-vpc-attachment \
    --transit-gateway-id $AWS_TRANSIT_GATEWAY_ID \
    --vpc-id $AWS_VPC_ID \
    --subnet-id $AWS_PRIVATE_SUBNET_ID

# create VPN connections
export AWS_VPN_CONNECTION_1=$(aws ec2 create-vpn-connection \
    --type ipsec.1 \
    --customer-gateway-id $AWS_CUSTOMER_GATEWAY_1 \
    --transit-gateway-id $AWS_TRANSIT_GATEWAY_ID \
    --options TunnelOptions= "**********"=$AWS_T1_IP.0/30,PreSharedKey=$SHARED_SECRET},{TunnelInsideCidr=$AWS_T2_IP.0/30,PreSharedKey=$SHARED_SECRET}]" \
    --query "" --output text)

export AWS_VPN_CONNECTION_2=$(aws ec2 create-vpn-connection \
    --type ipsec.1 \
    --customer-gateway-id $AWS_CUSTOMER_GATEWAY_2 \
    --transit-gateway-id $AWS_TRANSIT_GATEWAY_ID \
    --options TunnelOptions= "**********"=$AWS_T3_IP.0/30,PreSharedKey=$SHARED_SECRET},{TunnelInsideCidr=$AWS_T4_IP.0/30,PreSharedKey=$SHARED_SECRET}]" \
    --query "" --output text)

# retrieve configuration of two VPN connections (Generic Vendor + ikev2)
export AWS_GW_IP_1=$(aws ec2 describe-vpn-connections --query "VpnConnections[0].Options.TunnelOptions[0].OutsideIpAddress" --output text)
export AWS_GW_IP_2=$(aws ec2 describe-vpn-connections --query "VpnConnections[0].Options.TunnelOptions[1].OutsideIpAddress" --output text)
export AWS_GW_IP_3=$(aws ec2 describe-vpn-connections --query "VpnConnections[1].Options.TunnelOptions[0].OutsideIpAddress" --output text)
export AWS_GW_IP_4=$(aws ec2 describe-vpn-connections --query "VpnConnections[1].Options.TunnelOptions[1].OutsideIpAddress" --output text)
export AWS_INT_IP_1=$(aws ec2 describe-vpn-connections --query "VpnConnections[0].Options.TunnelOptions[0].TunnelInsideCidr" --output text)
export AWS_INT_IP_2=$(aws ec2 describe-vpn-connections --query "VpnConnections[0].Options.TunnelOptions[1].TunnelInsideCidr" --output text)
export AWS_INT_IP_3=$(aws ec2 describe-vpn-connections --query "VpnConnections[1].Options.TunnelOptions[0].TunnelInsideCidr" --output text)
export AWS_INT_IP_4=$(aws ec2 describe-vpn-connections --query "VpnConnections[1].Options.TunnelOptions[1].TunnelInsideCidr" --output text)

# print out configs for sanity checking later
cat << EOF

----------------------------------------------------------------
VPN Gateway Configs:
Gateway 1:
- Customer GW IP: $INTERFACE_0_IP_ADDRESS
- Tunnel 1: External: $AWS_GW_IP_1 / Internal: $AWS_INT_IP_1
- Tunnel 2: External: $AWS_GW_IP_2 / Internal: $AWS_INT_IP_2
Gateway 2:
- Customer GW IP: $INTERFACE_1_IP_ADDRESS 
- Tunnel 1: $AWS_GW_IP_3 / Internal: $AWS_INT_IP_3
- Tunnel 2: $AWS_GW_IP_4 / Internal: $AWS_INT_IP_4
-----------------------------------------------------------------
EOF


############################################################################
# EC2 TEST INSTANCE
# - https://www.geeksforgeeks.org/launching-an-ec2-instance-using-aws-cli/
############################################################################

export AWS_KEY_NAME="aws-keypair"
export AWS_KEY_FILE="aws-keypair.pem"
export SECURITY_GROUP_NAME="vpn-access"
export AMI_ID="ami-01f87c43e618bf8f0"  # Ubuntu 20.04 LTS amd64

# create keypair
aws ec2 create-key-pair --key-name $AWS_KEY_NAME --query "KeyMaterial" \
    --output text > $AWS_KEY_FILE

# create security group
export AWS_SECURITY_GROUP_ID=$(aws ec2 create-security-group --group-name $SECURITY_GROUP_NAME \
    --description $SECURITY_GROUP_NAME \
    --vpc-id $AWS_VPC_ID \
    --query GroupId --output text)

# authorize access for security group (from anywhere for simplicity)
aws ec2 authorize-security-group-ingress --group-id $AWS_SECURITY_GROUP_ID \
    --protocol tcp --port 22 --cidr $GCP_RANGE # SSH only from my GCP project
aws ec2 authorize-security-group-ingress --group-id $AWS_SECURITY_GROUP_ID \
    --protocol tcp --port 80 --cidr 0.0.0.0/0 # access HTTP server from anywhere
aws ec2 authorize-security-group-ingress --group-id $AWS_SECURITY_GROUP_ID \
    --protocol icmp --port all --cidr 0.0.0.0/0 # ping server from anywhere

# launch instance
export AWS_EC2_INSTANCE_ID=$(aws ec2 run-instances --image-id $AMI_ID \
    --count 1 \
    --instance-type t2.micro \
    --key-name $AWS_KEY_NAME \
    --security-group-ids $AWS_SECURITY_GROUP_ID \
    --subnet-id $AWS_PRIVATE_SUBNET_ID \
    --query "Instances[0].InstanceId" --output text)

# get instance IP address
export AWS_EC2_INSTANCE_IP=$(aws ec2 describe-instances --instance-id $AWS_EC2_INSTANCE_ID \
    --query "Reservations[].Instances[].PrivateIpAddress" --output text)


############################################################################
# GCP TUNNELS + ROUTER
# (TIP) verify matching IPs in both GCP and AWS consoles if BGP session issues
# ----------------------------------------------------------------
# VPN Gateway Configs:
# Gateway 1:
# - Customer GW IP: 35.242.55.180
# - Tunnel 1: External: 13.52.40.49 / Internal: 169.254.202.0/30
# - Tunnel 2: External: 52.52.225.196 / Internal: 169.254.203.0/30
# Gateway 2:
# - Customer GW IP: 35.220.54.143 
# - Tunnel 1: 52.9.126.13 / Internal: 169.254.200.0/30
# - Tunnel 2: 54.219.92.171 / Internal: 169.254.201.0/30
# -----------------------------------------------------------------
############################################################################

export PEER_GATEWAY_NAME="aws-vpn-gateway"
export IKE_VERSION="2"

# create external VPN gateway
gcloud compute external-vpn-gateways create $PEER_GATEWAY_NAME \
    --interfaces 0=$AWS_GW_IP_1,1=$AWS_GW_IP_2,2=$AWS_GW_IP_3,3=$AWS_GW_IP_4

# create four VPN tunnels
gcloud compute vpn-tunnels create tunnel-1 \
    --peer-external-gateway $PEER_GATEWAY_NAME \
    --peer-external-gateway-interface 0 \
    --region $GCP_REGION \
    --ike-version $IKE_VERSION \
    --shared-secret $SHARED_SECRET \
    --router $ROUTER_NAME \
    --vpn-gateway $VPN_GATEWAY_NAME \
    --interface 0

gcloud compute vpn-tunnels create tunnel-2 \
    --peer-external-gateway $PEER_GATEWAY_NAME \
    --peer-external-gateway-interface 1 \
    --region $GCP_REGION \
    --ike-version $IKE_VERSION \
    --shared-secret $SHARED_SECRET \
    --router $ROUTER_NAME \
    --vpn-gateway $VPN_GATEWAY_NAME \
    --interface 0

gcloud compute vpn-tunnels create tunnel-3 \
    --peer-external-gateway $PEER_GATEWAY_NAME \
    --peer-external-gateway-interface 2 \
    --region $GCP_REGION \
    --ike-version $IKE_VERSION \
    --shared-secret $SHARED_SECRET \
    --router $ROUTER_NAME \
    --vpn-gateway $VPN_GATEWAY_NAME \
    --interface 1

gcloud compute vpn-tunnels create tunnel-4 \
    --peer-external-gateway $PEER_GATEWAY_NAME \
    --peer-external-gateway-interface 3 \
    --region $GCP_REGION \
    --ike-version $IKE_VERSION \
    --shared-secret $SHARED_SECRET \
    --router $ROUTER_NAME \
    --vpn-gateway $VPN_GATEWAY_NAME \
    --interface 1

# add four cloud router interfaces (use string parameter substitution with .2 for router)
gcloud compute routers add-interface $ROUTER_NAME \
    --interface-name int-1 \
    --vpn-tunnel tunnel-1 \
    --ip-address "${AWS_INT_IP_1/.0\/30/.2}" \
    --mask-length 30 \
    --region $GCP_REGION

gcloud compute routers add-interface $ROUTER_NAME \
    --interface-name int-2 \
    --vpn-tunnel tunnel-2 \
    --ip-address "${AWS_INT_IP_2/.0\/30/.2}" \
    --mask-length 30 \
    --region $GCP_REGION

gcloud compute routers add-interface $ROUTER_NAME \
    --interface-name int-3 \
    --vpn-tunnel tunnel-3 \
    --ip-address "${AWS_INT_IP_3/.0\/30/.2}" \
    --mask-length 30 \
    --region $GCP_REGION

gcloud compute routers add-interface $ROUTER_NAME \
    --interface-name int-4 \
    --vpn-tunnel tunnel-4 \
    --ip-address "${AWS_INT_IP_4/.0\/30/.2}" \
    --mask-length 30 \
    --region $GCP_REGION

# add four BGP peers (use string parameter substitution with .1 for peer)
gcloud compute routers add-bgp-peer $ROUTER_NAME \
    --peer-name aws-conn1-tunn1 \
    --peer-asn $ASN_AWS \
    --interface int-1 \
    --peer-ip-address "${AWS_INT_IP_1/.0\/30/.1}" \
    --region $GCP_REGION

gcloud compute routers add-bgp-peer $ROUTER_NAME \
    --peer-name aws-conn1-tunn2 \
    --peer-asn $ASN_AWS \
    --interface int-2 \
    --peer-ip-address "${AWS_INT_IP_2/.0\/30/.1}" \
    --region $GCP_REGION

gcloud compute routers add-bgp-peer $ROUTER_NAME \
    --peer-name aws-conn1-tunn3 \
    --peer-asn $ASN_AWS \
    --interface int-3 \
    --peer-ip-address "${AWS_INT_IP_3/.0\/30/.1}" \
    --region $GCP_REGION

gcloud compute routers add-bgp-peer $ROUTER_NAME \
    --peer-name aws-conn1-tunn4 \
    --peer-asn $ASN_AWS \
    --interface int-4 \
    --peer-ip-address "${AWS_INT_IP_4/.0\/30/.1}" \
    --region $GCP_REGION

# verify
gcloud compute routers get-status $ROUTER_NAME \
    --region $GCP_REGION \
    --format='flattened(result.bgpPeerStatus[].name, result.bgpPeerStatus[].ipAddress, result.bgpPeerStatus[].peerIpAddress)'


############################################################################
# GCP COMPUTE ENGINE TEST INSTANCE
############################################################################

export GCP_INSTANCE_ID="gcp-test-instance"

# create compute engine instance (with no external IP)
gcloud compute instances create $GCP_INSTANCE_ID \
    --project=$PROJECT_ID \
    --zone=$GCP_ZONE \
    --machine-type=e2-micro \
    --network-interface=subnet=$SUBNET_NAME,no-address \
    --metadata=enable-oslogin=TRUE \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --image-project debian-cloud \
    --image-family debian-11

# enable firewall access on port 22
gcloud compute firewall-rules create --network=$NETWORK_NAME test-allow-ssh --allow=tcp:22

# copy AWS keypair file to instance
gcloud compute scp ./$AWS_KEY_FILE $GCP_INSTANCE_ID:~

# test a ping of EC2 instance
gcloud compute ssh --zone $GCP_ZONE $GCP_INSTANCE_ID \
    --tunnel-through-iap \
    --command "ping $AWS_EC2_INSTANCE_IP"
ANCE_ID \
    --tunnel-through-iap \
    --command "ping $AWS_EC2_INSTANCE_IP"
