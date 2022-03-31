#date: 2022-03-31T16:59:53Z
#url: https://api.github.com/gists/ac377b4e3d0d23e0d5369a10ff1b4696
#owner: https://api.github.com/users/lesliejsmith

import boto3
import boto3.session


# #download your keys and set up a new section in credentials called permanent
my_session = boto3.session.Session(profile_name="permanent")

mfa_serial = my_session._session.full_config['profiles']['permanent']['mfa_serial']
mfa_token = input('Please enter your 6 digit MFA code:')
client = my_session.client('sts')
MFA_validated_token = client.get_session_token( DurationSeconds=5000,
                                            SerialNumber=mfa_serial, TokenCode=mfa_token)
