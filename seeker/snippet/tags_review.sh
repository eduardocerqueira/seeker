#date: 2023-01-25T16:57:47Z
#url: https://api.github.com/gists/2e1ec8c0e90cc63d5e7ad8dab06ff972
#owner: https://api.github.com/users/Nicolas-Richard

# Utility script to identify AWS resources missing tags by parsing the TF state file.

# vision for this tool :
# produce a report of missing tags per workspace




# Invoke like this to display the untagged resources :
# tags_review.sh 

# Invoke like this to get the count of untagged resources :
# tags_review.sh | jq --slurp length

# Invoke like this to display the count of untagged resources per resource type :
# tags_review.sh | grep '"type"' | sort | uniq -c | sort -rn 


# terraform init && terraform state pull   \

cat ~/Downloads/sv-6TUAthRMFCbHTmes.tfstate  \
 | jq '.resources[]

# selecting aws resources only
 | select(.type|test("aws_")) 

# filter out data sources (the alternative is mode=data)
 | select(.mode|test("managed"))


 # filter out AWS resources that cant be tagged
 # (this is a blacklist approach, but we could also make a whitelist of the resources we absolutely want to tag to focus our effort)
 | select(.type|test("aws_acm_certificate_validation") | not)
 
 | select(.type|test("aws_api_gateway_authorizer") | not)
 | select(.type|test("aws_api_gateway_deployment") | not)
 | select(.type|test("aws_api_gateway_integration") | not)
 | select(.type|test("aws_api_gateway_integration_response") | not)
 | select(.type|test("aws_api_gateway_method") | not)
 | select(.type|test("aws_api_gateway_method_response") | not)
 | select(.type|test("aws_api_gateway_method_settings") | not)
 | select(.type|test("aws_api_gateway_model") | not)
 | select(.type|test("aws_api_gateway_resource") | not)

 | select(.type|test("aws_cloudfront_origin_access_identity") | not)
 | select(.type|test("aws_cloudfront_response_headers_policy") | not)
 | select(.type|test("aws_cloudwatch_log_subscription_filter") | not)
 
 | select(.type|test("aws_dynamodb_table_item") | not)
 
 | select(.type|test("aws_ecr_lifecycle_policy") | not) 
 | select(.type|test("aws_ecr_replication_configuration") | not)
 | select(.type|test("aws_ecr_repository_policy") | not)
 
 | select(.type|test("aws_iam_group") | not)
 | select(.type|test("aws_iam_group_policy_attachment") | not)
 | select(.type|test("aws_iam_policy_document") | not)
 | select(.type|test("aws_iam_role_policy") | not)
 | select(.type|test("aws_iam_role_policy_attachment") | not)
 | select(.type|test("aws_iam_user_policy") | not)
 
 | select(.type|test("aws_kms_alias") | not)
 
 | select(.type|test("aws_lambda_function_url") | not)
 | select(.type|test("aws_lambda_permission") | not)

 | select(.type|test("aws_route53_query_log") | not)
 | select(.type|test("aws_route53_record") | not)

 | select(.type|test("aws_s3_bucket_metric") | not)
 | select(.type|test("aws_s3_bucket_notification") | not)
 | select(.type|test("aws_s3_bucket_policy") | not)
 | select(.type|test("aws_s3_bucket_public_access_block") | not)

 | select(.type|test("aws_security_group_rule") | not)

 | select(.type|test("aws_sns_topic_subscription") | not)

 | select(.type|test("aws_sqs_queue_policy") | not)

 | select(.type|test("aws_ssoadmin_account_assignment") | not)
 | select(.type|test("aws_ssoadmin_managed_policy_attachment") | not)
 | select(.type|test("aws_ssoadmin_permission_set_inline_policy") | not)


 # creating the output
 | { name: .name, type: .type, module: .module, tags: .instances[].attributes.tags } 

 # filtering the output for resources missing desired tag
 | select( any(.tags | has("Team"); not))  '


