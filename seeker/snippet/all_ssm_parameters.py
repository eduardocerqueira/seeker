#date: 2022-01-04T17:11:15Z
#url: https://api.github.com/gists/9724effaccf85ebb22405c6b3f56ac37
#owner: https://api.github.com/users/alkuzad

def ssm_parameters():
    ssm_path = "/example/path/" # / at the end

    ssm = boto3.client('ssm', region_name='eu-west-1')
    parameters = ssm.get_parameters_by_path(Path=ssm_path, WithDecryption=True)[
        'Parameters'
    ]
    return {x['Name'].replace(ssm_path, ""): x['Value'] for x in parameters}
    