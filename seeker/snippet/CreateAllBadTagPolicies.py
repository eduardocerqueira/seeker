#date: 2023-01-09T16:49:15Z
#url: https://api.github.com/gists/b98322477cdf70b32520cfb830f8f927
#owner: https://api.github.com/users/HumanSolutions


AWSServicesArray = "**********"

TagName = "MyTag"
TagValue = "MyTagsValue"

AllPoliciesArray = ["policies:\r\n\r\n"]

template = """
- name: aws-<SERVICE_HERE>-tag-compliance
  resource: <SERVICE_HERE>
  filters:
    - "tag:<TAGNAME_HERE>": "<TAGVALUE_HERE>"
"""

for Service in AWSServicesArray:
    ServicePolicy = template
    ServicePolicy = ServicePolicy.replace("<SERVICE_HERE>",Service)
    ServicePolicy = ServicePolicy.replace("<TAGNAME_HERE>",TagName)
    ServicePolicy = ServicePolicy.replace("<TAGVALUE_HERE>",TagValue)

    AllPoliciesArray.append(ServicePolicy)


AllPolicies = "\r\n\r\n".join(AllPoliciesArray)

print(AllPolicies)




arget","eks","elasticbeanstalk","elasticbeanstalk-environment","elasticsearch","elb","emr","eni","event-rule","event-rule-target","firehose","fsx","fsx-backup","gamelift-build","gamelift-fleet","glacier","glue-connection","glue-dev-endpoint","health-event","healthcheck","hostedzone","hsm","hsm-client","hsm-hapg","iam-certificate","iam-group","iam-policy","iam-profile","iam-role","iam-user","identity-pool","internet-gateway","iot","kafka","key-pair","kinesis","kinesis-analytics","kms","kms-key","lambda","lambda-layer","launch-config","launch-template-version","lightsail-db","lightsail-elb","lightsail-instance","log-group","message-broker","ml-model","nat-gateway","network-acl","network-addr","opswork-cm","opswork-stack","peering-connection","r53domain","rds","rds-cluster","rds-cluster-param-group","rds-cluster-snapshot","rds-param-group","rds-snapshot","rds-subnet-group","rds-subscription","redshift","redshift-snapshot","redshift-subnet-group","rest-account","rest-api","rest-resource","rest-stage","rest-vpclink","route-table","rrset","s3","sagemaker-endpoint","sagemaker-endpoint-config","sagemaker-job","sagemaker-model","sagemaker-notebook","sagemaker-transform-job","secrets-manager","security-group","shield-attack","shield-protection","simpledb","snowball","snowball-cluster","sns","sqs","ssm-activation","ssm-managed-instance","ssm-parameter","step-machine","storage-gateway","streaming-distribution","subnet","support-case","transit-attachment","transit-gateway","user-pool","vpc","vpc-endpoint","vpn-connection","vpn-gateway","waf","waf-regional"]

TagName = "MyTag"
TagValue = "MyTagsValue"

AllPoliciesArray = ["policies:\r\n\r\n"]

template = """
- name: aws-<SERVICE_HERE>-tag-compliance
  resource: <SERVICE_HERE>
  filters:
    - "tag:<TAGNAME_HERE>": "<TAGVALUE_HERE>"
"""

for Service in AWSServicesArray:
    ServicePolicy = template
    ServicePolicy = ServicePolicy.replace("<SERVICE_HERE>",Service)
    ServicePolicy = ServicePolicy.replace("<TAGNAME_HERE>",TagName)
    ServicePolicy = ServicePolicy.replace("<TAGVALUE_HERE>",TagValue)

    AllPoliciesArray.append(ServicePolicy)


AllPolicies = "\r\n\r\n".join(AllPoliciesArray)

print(AllPolicies)




