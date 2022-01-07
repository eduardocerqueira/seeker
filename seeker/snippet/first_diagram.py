#date: 2022-01-07T16:54:29Z
#url: https://api.github.com/gists/0bd93013b95d8ca47a35da6d64f07c5f
#owner: https://api.github.com/users/arnab44

from diagrams import Diagram
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.aws.network import ELB
from diagrams.aws.integration import SQS
from  diagrams.aws.storage import S3
from diagrams.saas.chat import Slack
with Diagram("mvc", show=True, direction="LR"):
    elb =  ELB("load balancer")
    service1 = EC2('Service1')
    service2 = EC2('Service2')
    service3 = EC2('service3')
    db = RDS("primary DB")
    
    elb >> service1 >> db
    elb >> service2 >> db
    service1 >> SQS('sqs') >> service3 >> S3('s3')
    service3 >> Slack('slack notification')