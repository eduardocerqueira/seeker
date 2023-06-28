#date: 2023-06-28T17:09:22Z
#url: https://api.github.com/gists/47ede52c48862866f732279167feb211
#owner: https://api.github.com/users/alphacrack

from diagrams import Cluster, Diagram, Node, Edge
from diagrams.aws.compute import EC2
from diagrams.aws.network import PrivateSubnet, RouteTable, VPC, APIGateway
from diagrams.aws.storage import S3
from diagrams.aws.database import Dynamodb
from diagrams.aws.management import Cloudwatch

with Diagram("AWS VPC with Gateway and Interface Endpoints", show=False):
    with Cluster("AWS Region"):
        with Cluster("Virtual Private Cloud"):
            vpc = VPC("VPC")
            gateway_endpoint = Node("Gateway")
            route_table = RouteTable("Route Table")

            with Cluster("Availability Zone 1"):
                subnet1 = PrivateSubnet("Private Subnet 1")
                ec2_1 = EC2("EC2 Instance 1")
                eni_1 = Node("Elastic Network Interface 1")
                subnet1 >> ec2_1 >> eni_1

            with Cluster("Availability Zone 2"):
                subnet2 = PrivateSubnet("Private Subnet 2")
                ec2_2 = EC2("EC2 Instance 2")
                eni_2 = Node("Elastic Network Interface 2")
                subnet2 >> ec2_2 >> eni_2

            vpc >> route_table >> gateway_endpoint

        services = [APIGateway("API Gateway"), Cloudwatch("Cloudwatch")]
        for service in services:
            eni_1 >> Edge(label="Interface Endpoint") >> service
            eni_2 >> Edge(label="Interface Endpoint") >> service

        s3 = S3("S3 Bucket")
        dynamodb = Dynamodb("DynamoDB")

        gateway_endpoint >> Edge(label="Gateway Endpoint") >> s3
        gateway_endpoint >> Edge(label="Gateway Endpoint") >> dynamodb
