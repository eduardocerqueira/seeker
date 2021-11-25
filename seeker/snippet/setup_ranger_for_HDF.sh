#date: 2021-11-25T17:00:35Z
#url: https://api.github.com/gists/e14055a5fc7accf4116258aa98ade129
#owner: https://api.github.com/users/gbandhalraj

# Run this script on Ranger host to 
# 1. create users in format $host.openstacklocal@apache.nifi e.g. abajwa-hdf-qe-bp-1.openstacklocal@apache.nifi
# 2. create Ranger policies for above Nifi users:
#  a) read policy for /flow 
#  b) read/write policies for /proxy


export admin=${admin:-nifiadmin}
export cluster=${cluster:-HDF}
export hosts=${hosts:-myhost1 myhost2 myhost3}
export realm=$realm
if [ -n "$realm" ]; then
    export realm=@$realm
fi

service="$cluster"_nifi
users="$admin $hosts"
for user in $users
do

	tee payload > /dev/null << EOF
{
    "name": "$user$realm",
    "password": "BadPass#1",
    "firstName":"$user",
    "lastName":"",
    "emailAddress":"",
    "status": "1",
    "userRoleList": ["ROLE_USER"],
    "groupIdList":["1"]
}
EOF
	curl -i -u admin:admin  -H 'Content-Type: application/json' -X POST  http://localhost:6080/service/xusers/secure/users -d @payload
	/bin/rm -f payload
	
done

echo "Attempting to create /* policy for $admin"

tee payload > /dev/null << EOF
{
	"policyType": "0",
	"name": "/*",
	"isEnabled": "true",	
    "isAuditEnabled": "true",
    "description": "",
    "resources":
    {
        "nifi-resource":
        {
            "values":["/*"],
            "isRecursive":"",
            "isExcludes":false
        }
    },
    "policyItems":
    [{
      "users":["$admin$realm"],
      "accesses":[{"type":"READ", "isAllowed":true}, {"type":"WRITE", "isAllowed":true}]  
    }],
    "denyPolicyItems":[],
    "allowExceptions":[],
    "denyExceptions":[],
    "service":"$service"
}
EOF

curl -i -u admin:admin  -H 'Content-Type: application/json' -X POST http://localhost:6080/service/plugins/policies -d @payload


users=""
for host in $hosts
do
	user="$host$realm"
	if [ -z "$users" ]
	then
		users=\"$user\"	
	else
		users=$users,\"$user\"
	fi
done
echo "Attempting to create /flow policy for $users"

tee payload > /dev/null << EOF
{
	"policyType": "0",
	"name": "/flow",
	"isEnabled": "true",	
    "isAuditEnabled": "true",
    "description": "",
    "resources":
    {
        "nifi-resource":
        {
            "values":["/flow"],
            "isRecursive":"",
            "isExcludes":false
        }
    },
    "policyItems":
    [{
      "users":[$users],
      "accesses":[{"type":"READ", "isAllowed":true}]  
    }],
    "denyPolicyItems":[],
    "allowExceptions":[],
    "denyExceptions":[],
    "service":"$service"
}
EOF

curl -i -u admin:admin  -H 'Content-Type: application/json' -X POST http://localhost:6080/service/plugins/policies -d @payload


echo "Attempting to create /proxy policy for $users"

tee payload > /dev/null << EOF
{
	"policyType": "0",
	"name": "/proxy",
	"isEnabled": "true",	
    "isAuditEnabled": "true",
    "description": "",
    "resources":
    {
        "nifi-resource":
        {
            "values":["/proxy"],
            "isRecursive":"",
            "isExcludes":false
        }
    },
    "policyItems":
    [{
      "users":[$users],
      "accesses":[{"type":"READ", "isAllowed":true}, {"type":"WRITE", "isAllowed":true}]  
    }],
    "denyPolicyItems":[],
    "allowExceptions":[],
    "denyExceptions":[],
    "service":"$service"
}
EOF

curl -i -u admin:admin  -H 'Content-Type: application/json' -X POST http://localhost:6080/service/plugins/policies -d @payload




tee payload > /dev/null << EOF
{
	"policyType": "0",
	"name": "/data/*",
	"isEnabled": "true",	
    "isAuditEnabled": "true",
    "description": "",
    "resources":
    {
        "nifi-resource":
        {
            "values":["/data/*"],
            "isRecursive":"",
            "isExcludes":false
        }
    },
    "policyItems":
    [{
      "users":[$users],
      "accesses":[{"type":"READ", "isAllowed":true}, {"type":"WRITE", "isAllowed":true}]  
    }],
    "denyPolicyItems":[],
    "allowExceptions":[],
    "denyExceptions":[],
    "service":"$service"
}
EOF

curl -i -u admin:admin  -H 'Content-Type: application/json' -X POST http://localhost:6080/service/plugins/policies -d @payload


