#date: 2022-04-19T16:59:57Z
#url: https://api.github.com/gists/be06fa44372c335ebefc1f6394ca9eed
#owner: https://api.github.com/users/khpraful

import json
import requests
import base64
import config

pending_approvals_base_url = 'https://vsrm.dev.azure.com/<organization>/<project>/_apis/release/approvals'
approve_base_url = 'https://vsrm.dev.azure.com/<organization>/<project/_apis/release/approvals/'

authorization = str(base64.b64encode(bytes(':'+config.pat, 'ascii')), 'ascii')
headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': 'Basic '+authorization
}

params = {
    'assignedToFilter': config.user
}

response = requests.get(url=pending_approvals_base_url, headers=headers, params=params)

approvalDetails=[]
approvalIds=[]

for i in response.json()['value']:
    approvalDetails.append([i['id'],i['releaseDefinition']['name'],i['releaseEnvironment']['name'], i['approvalType'],i['approver']['uniqueName']])
    approvalIds.append(i['id'])

print('====================================================================================================')
print('Approval List:')
print('Approval Id, Release, Environment, Approval Type, Approver')
for j in approvalDetails:
    print(j)
print('====================================================================================================')
print ('Approval Ids:')
print(approvalIds)
print('====================================================================================================')
approvals = input("Enter approval ids for the release requests you want to approve (as comma-separated values):")
print('====================================================================================================')
approval_list = approvals.split(",") if approvals else []
if approval_list==[]:
    print("No approval ids provided. Operation cancelled!!!")
else:
    confirm = input("Are you sure you want to approve specified release requests(y/n)?")
    print('====================================================================================================')
    if (confirm.lower()=='y' or confirm.lower()=='yes'):
        comments=input("Enter approval comments:")
        print('====================================================================================================')
        for i in approval_list:
            data = {
                    'status': 'approved',
                    'comments': comments
                }
            response = requests.patch(url=approve_base_url + str(i) +'?api-version=6.0', headers=headers, data=json.dumps(data))
            if response.status_code==200:
                print('Deployment request with id ' + i + ' approved successfully.')
            elif response.status_code==401:
                print('You are not authorized to perform this operation. Please check your permissions.')
            elif response.status_code == 400:
                print('Bad Request, unable to proceed. Please check approval settings for this stage.')
    else:
        print('====================================================================================================')
        print("Operation cancelled!!!")
