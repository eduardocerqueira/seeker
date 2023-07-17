#date: 2023-07-17T16:51:26Z
#url: https://api.github.com/gists/5d4351fef8b60270dc0d444bb675e796
#owner: https://api.github.com/users/Rajchowdhury420

import boto3
from datetime import datetime

# Specify your AWS account ID and the names of the roles to be deleted
account_id = '653753205712'
role_names = ['steveRole', 's3role', 'testLamdaFormS3', 'AutomationServiceRole']

# Create an IAM client
iam = boto3.client('iam')

# Iterate over the role names and delete each role
for role_name in role_names:
    try:
        # Delete the IAM role
        response = iam.delete_role(RoleName=role_name)
        print(f"Deleted role: {role_name}")

        # Log the deletion with current date and time
        deletion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Deletion time: {deletion_time}")
    except iam.exceptions.NoSuchEntityException:
        print(f"Role does not exist: {role_name}")
    except Exception as e:
        print(f"Failed to delete role: {role_name}. Error: {str(e)}")
