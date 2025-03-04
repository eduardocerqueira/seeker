#date: 2025-03-04T17:06:43Z
#url: https://api.github.com/gists/929e2738a27bbbf151bf4d9bdad7c640
#owner: https://api.github.com/users/stephenVertex

#!/bin/bash
set -e

echo "Retrieving configuration values from AWS CloudFormation outputs..."

# Get stack name from samconfig.toml
STACK_NAME=$(grep "stack_name" samconfig.toml | head -1 | cut -d'=' -f2 | tr -d ' ' | tr -d '"')
if [ -z "$STACK_NAME" ]; then
  echo "Error: Stack name not found in samconfig.toml"
  exit 1
fi

echo "Using CloudFormation stack: $STACK_NAME"

# Get outputs from CloudFormation
echo "Retrieving outputs from CloudFormation stack..."
CF_OUTPUTS=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --query "Stacks[0].Outputs" --output json)

# Extract values from outputs
echo "Extracting configuration values..."
FUNCTION_URL=$(echo $CF_OUTPUTS | jq -r '.[] | select(.OutputKey=="ImgGenerateLambdaUrl") | .OutputValue')
BUCKET_NAME=$(echo $CF_OUTPUTS | jq -r '.[] | select(.OutputKey=="BucketName") | .OutputValue')
BUCKET_URL=$(echo $CF_OUTPUTS | jq -r '.[] | select(.OutputKey=="BucketURL") | .OutputValue')

# Fallbacks for missing values
if [ "$FUNCTION_URL" == "null" ] || [ -z "$FUNCTION_URL" ]; then
  echo "Warning: Lambda function URL not found in CloudFormation outputs."
  FUNCTION_URL="Function URL not found"
else
  echo "Found Lambda function URL: $FUNCTION_URL"
fi

if [ "$BUCKET_NAME" == "null" ] || [ -z "$BUCKET_NAME" ]; then
  echo "Warning: Bucket name not found in CloudFormation outputs."
  BUCKET_NAME="codegen-images-public"
  echo "Using default bucket name: $BUCKET_NAME"
else
  echo "Found S3 bucket name: $BUCKET_NAME"
fi

if [ "$BUCKET_URL" == "null" ] || [ -z "$BUCKET_URL" ]; then
  echo "Warning: Bucket URL not found in CloudFormation outputs."
  BUCKET_URL="https://${BUCKET_NAME}.s3.amazonaws.com"
  echo "Using constructed bucket URL: $BUCKET_URL"
else
  echo "Found S3 bucket URL: $BUCKET_URL"
fi

# Get the actual function name 
FUNCTION_NAME=$(aws cloudformation describe-stack-resources --stack-name $STACK_NAME --query "StackResources[?ResourceType=='AWS::Lambda::Function'].PhysicalResourceId" --output text)
if [ -z "$FUNCTION_NAME" ]; then
  echo "Warning: Lambda function name not found in CloudFormation resources."
  FUNCTION_NAME="img-generate-lambda"
  echo "Using default function name: $FUNCTION_NAME"
else
  echo "Found Lambda function name: $FUNCTION_NAME"
fi

# Create .env file
echo "Writing configuration to .env file..."
cat > .env << EOF
LAMBDA_FUNCTION_NAME=${FUNCTION_NAME}
LAMBDA_FUNCTION_URL=${FUNCTION_URL}
S3_BUCKET_NAME=${BUCKET_NAME}
S3_BUCKET_URL=${BUCKET_URL}
EOF

echo "Configuration written to .env file successfully!"
echo "You can now use these environment variables in your application." 