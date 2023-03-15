#date: 2023-03-15T16:57:25Z
#url: https://api.github.com/gists/58481f70ad11155e11202c0cbbe1ec79
#owner: https://api.github.com/users/carlosspohr

#/bin/bash

# @author Carlos Spohr
# Este script cria o bucket de produção e replicação na AWS juntamente da
# iam role, role policies para replicação.
# Para rodar este script você precisará ter o aws cli instalado e com uma chave(key) 
# que permita você criar estes itens via aws cli.
# Este script é feito pra linux e depende apenas do pacote jq. Você pode instalar ele assim:
# CentOS: yum install -y jq
# Ubuntu/debian: apt-get install -y jq

# Usage:
# Não precisa ser sudo.
# create-bucket.sh bucket-name

BUCKET=$1;
BUCKET_REPLICATION="$1-replication";

if [ -z $BUCKET ]; then
	echo -e "\e[1;31m=============================================================================================\e[0m";
	echo -e "\e[1;31m                         O NOME DO BUCKET NÃO FOI INFORMADO!\e[0m";
	echo -e "\e[1;31m=============================================================================================\e[0m";
    exit 2;
fi

echo "creating bucket $BUCKET";
echo "creating replication bucket $BUCKET_REPLICATION";

aws s3 mb s3://$BUCKET --region sa-east-1
aws s3 mb s3://$BUCKET_REPLICATION --region us-east-1

echo "Adding versioning";

aws s3api put-bucket-versioning --bucket $BUCKET --versioning-configuration MFADelete=Disabled,Status=Enabled
aws s3api put-bucket-versioning --bucket $BUCKET_REPLICATION --versioning-configuration MFADelete=Disabled,Status=Enabled

echo "Creating iam role for replication";

ROLE_FILE="$BUCKET.json";

/bin/rm -f $ROLE_FILE

cat > $ROLE_FILE <<- EOM
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "s3.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOM

IAM_ROLE_NAME="s3crr_role_for_$BUCKET";

aws iam create-role --role-name $IAM_ROLE_NAME --assume-role-policy-document file://$ROLE_FILE

echo "Creating role permissions for replication";

ROLE_PERMISSIONS_FILE="$BUCKET.json";

/bin/rm -f $ROLE_PERMISSIONS_FILE

cat > $ROLE_PERMISSIONS_FILE <<- EOM
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObjectVersionForReplication",
        "s3:GetObjectVersionAcl",
        "s3:GetObjectVersionTagging"
      ],
      "Resource": ["arn:aws:s3:::$BUCKET/*"] 
    },
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket", "s3:GetReplicationConfiguration"],
      "Resource": ["arn:aws:s3:::$BUCKET"]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:ReplicateObject",
        "s3:ReplicateDelete",
        "s3:ReplicateTags"
      ],
      "Resource": "arn:aws:s3:::$BUCKET_REPLICATION/*" 
    }
  ]
}
EOM

IAM_ROLE_POLICY_NAME="s3crr_role_policy_for_$BUCKET";

aws iam put-role-policy --role-name $IAM_ROLE_NAME --policy-document file://$ROLE_PERMISSIONS_FILE --policy-name $IAM_ROLE_POLICY_NAME

echo "Adding replication metric to bucket $BUCKET";

REPLICATION_FILE="$BUCKET.json";

# Obtenho o arn:role name recem criada.
ROLE_ARN=$(aws iam get-role --role-name $IAM_ROLE_NAME | jq '.Role.Arn')

/bin/rm -f $REPLICATION_FILE

cat > $REPLICATION_FILE <<- EOM
{
    "Role": $ROLE_ARN,
    "Rules": [
        {
            "ID": "$BUCKET_REPLICATION-rule",
            "Status": "Enabled",
            "Priority": 0,
            "DeleteMarkerReplication": { "Status": "Disabled" },
            "Filter" : {},
            "Destination": {
                "Bucket": "arn:aws:s3:::$BUCKET_REPLICATION"
            }
        }
    ]
}
EOM

aws s3api put-bucket-replication --bucket $BUCKET --replication-configuration file://$REPLICATION_FILE

echo "Done!";

exit 1;