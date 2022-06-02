#date: 2022-06-02T17:18:02Z
#url: https://api.github.com/gists/75a7543b22034d9e46ebfe0495ce8fea
#owner: https://api.github.com/users/willmitchell

# watch the demo at https://youtu.be/cGUNf1FMNvI

KUBECTL_URL='https://amazon-eks.s3.us-west-2.amazonaws.com/1.20.4/2021-04-12/bin/linux/amd64/kubectl'
AWS_CLI_V2_URL='https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip'
CRED_PROCESS_URL='https://raw.githubusercontent.com/pahud/vscode/main/.devcontainer/bin/aws-sso-credential-process'

DEFAULT_PROFILE='default'
DEFAULT_REGION='ap-northeast-1'
DEFAULT_OUTPUT='json'
SSO_START_URL='https://pahud-sso.awsapps.com/start'
SSO_REGION='us-east-1'
SSO_ACCOUNT_ID='123456789012'
SSO_ROLE_NAME='AdministratorAccess'

PROJECT_DIR="${PWD}"
WORKSPACE_BIN='/workspace/bin'

export PATH=${WORKSPACE_BIN}:${PATH}

cd $HOME

# install aws-cli v2
curl "${AWS_CLI_V2_URL}" -o "awscliv2.zip" && \
  unzip awscliv2.zip && \
  ./aws/install -i /workspace/aws-cli -b ${WORKSPACE_BIN}
alias aws="${WORKSPACE_BIN}/aws"

# install kubectl
curl -o kubectl "${KUBECTL_URL}" && \
  chmod +x kubectl && \
  mv kubectl ${WORKSPACE_BIN}

cd ${WORKSPACE_BIN} && \
curl -o aws-sso-credential-process "${CRED_PROCESS_URL}" && \
chmod +x aws-sso-credential-process && \
cd ${PROJECT_DIR}

aws configure set credential_process ${WORKSPACE_BIN}/aws-sso-credential-process
touch ~/.aws/credentials && chmod 600 $_

echo "generate the ~/.aws/config"

cat << EOF > ~/.aws/config
[${DEFAULT_PROFILE}]
credential_process = ${WORKSPACE_BIN}/aws-sso-credential-process
sso_start_url = ${SSO_START_URL}
sso_region = ${SSO_REGION}
sso_account_id = ${SSO_ACCOUNT_ID}
sso_role_name =${SSO_ROLE_NAME}
region = ${DEFAULT_REGION}
output = ${DEFAULT_OUTPUT}
EOF

# skip the configuration as we just generated the config above
#aws configure sso --profile default

# login to authenticate again
aws sso login



