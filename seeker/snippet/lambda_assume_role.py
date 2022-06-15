#date: 2022-06-15T16:55:31Z
#url: https://api.github.com/gists/8aa091818ad33e234a37d0971dbb2924
#owner: https://api.github.com/users/sf-walsh

import boto3

# you can assign role in the function like below
# ROLE_ARN = 'arn:aws:iam::01234567890:role/my_role'
#
# or you can pass role as an evironment varibale
# ROLE_ARN = os.environ['role_arn']

ROLE_ARN = = os.environ['role_arn']

def aws_session(role_arn=None, session_name='my_session'):
    """
    If role_arn is given assumes a role and returns boto3 session
    otherwise return a regular session with the current IAM user/role
    """
    if role_arn:
        client = boto3.client('sts')
        response = client.assume_role(RoleArn=role_arn, RoleSessionName=session_name)
        session = boto3.Session(
            aws_access_key_id=response['Credentials']['AccessKeyId'],
            aws_secret_access_key=response['Credentials']['SecretAccessKey'],
            aws_session_token=response['Credentials']['SessionToken'])
        return session
    else:
        return boto3.Session()

def lambda_handler(event, context):
    session_assumed = aws_session(role_arn=ROLE_ARN, session_name='my_lambda')
    session_regular = aws_session()
  
    print(session_assumed.client('sts').get_caller_identity()['Account'])
    print(session_regular.client('sts').get_caller_identity()['Account'])
