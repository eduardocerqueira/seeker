#date: 2022-03-25T17:08:09Z
#url: https://api.github.com/gists/19f82994932d41572a02e7cb946fed37
#owner: https://api.github.com/users/JBOClara

# Generate a plan
terraform plan -out=/tmp/heres-the-plan.txt
# Convert to json
terraform show -json /tmp/heres-the-plan.txt | jq '.' > /tmp/heres-the-plan.json

# extract values.name from record with an address starting with aws_route53_record 
# example: {
#          "address": "aws_route53_record.www_exampledomain_de_TXT",
#          "mode": "managed",
#          "type": "aws_route53_record",
#          "name": "www_exampledomain_de_TXT",
#          "provider_name": "registry.terraform.io/hashicorp/aws",
#          "schema_version": 2,
#          "values": {
#            "alias": [],
#            "failover_routing_policy": [],
#            "geolocation_routing_policy": [],
#            "health_check_id": null,
#            "latency_routing_policy": [],
#            "multivalue_answer_routing_policy": null,
#            "name": "www.exampledomain.de",
#            "records": [
#              "3|welcome",
#              "l|fr"
#            ],
#            "set_identifier": null,
#            "ttl": 3600,
#            "type": "TXT",
#            "weighted_routing_policy": [],
#            "zone_id": "Z0YYYYYYYYYYYYYU5"
#          },
#          "sensitive_values": {
#            "alias": [],
#            "failover_routing_policy": [],
#            "geolocation_routing_policy": [],
#            "latency_routing_policy": [],
#            "records": [
#              false,
#              false
#            ],
#            "weighted_routing_policy": []
#          }
#        }, 

# The -var-file is optional according to your context.
cat /tmp/heres-the-plan.json| jq '.planned_values.root_module.resources[] | select(.type == "aws_route53_record") | "terraform import -var-file=./.secret-clear.json " + .address + " " + .values.zone_id + "_" + .values.name + "_" + .values.type' 