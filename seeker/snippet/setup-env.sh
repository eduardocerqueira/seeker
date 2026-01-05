#date: 2026-01-05T17:00:53Z
#url: https://api.github.com/gists/9f96bef6b07b40f18fd3b7ea4c4ba673
#owner: https://api.github.com/users/chase-robbins

#!/bin/bash

# setup-env.sh - Generate .env file from CloudFormation stack outputs
# Usage: ./scripts/setup-env.sh <environment-name>

set -e

echo "DEBUG: Script started"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "DEBUG: SCRIPT_DIR=$SCRIPT_DIR"

# Change to the project root (parent directory of scripts)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
echo "DEBUG: PROJECT_ROOT=$PROJECT_ROOT"

cd "$PROJECT_ROOT"
echo "DEBUG: Changed to project root, pwd=$(pwd)"

if [ $# -eq 0 ]; then
    echo "Usage: ./scripts/setup-env.sh <environment-name>"
    echo "Example: ./scripts/setup-env.sh prod (will query stack: prod-kubera)"
    echo "Can be run from any directory"
    exit 1
fi

STACK_NAME="$1-kubera"
API_ENV_FILE="src/assets/api/.env"
FRONTEND_ENV_FILE="src/assets/frontend/.env.local"
ALGOLIA_SYNC_ENV_FILE="src/assets/lambda/algolia-sync/.env"

echo "DEBUG: STACK_NAME=$STACK_NAME"
echo "DEBUG: API_ENV_FILE=$API_ENV_FILE"
echo "DEBUG: FRONTEND_ENV_FILE=$FRONTEND_ENV_FILE"
echo "DEBUG: ALGOLIA_SYNC_ENV_FILE=$ALGOLIA_SYNC_ENV_FILE"

echo "🔍 Fetching CloudFormation outputs for stack: $STACK_NAME"

# Check if AWS CLI is available
echo "DEBUG: Checking if AWS CLI is available..."
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed or not in PATH"
    exit 1
fi
echo "DEBUG: AWS CLI found at $(which aws)"

# Check AWS credentials
echo "DEBUG: Checking AWS credentials..."
CALLER_IDENTITY=$(aws sts get-caller-identity 2>&1)
if [ $? -eq 0 ]; then
    echo "DEBUG: AWS credentials valid"
    echo "DEBUG: Identity: $(echo "$CALLER_IDENTITY" | jq -r '.Arn')"
else
    echo "DEBUG: AWS credentials check failed - may need to run 'awslogin'"
    echo "DEBUG: Error: $CALLER_IDENTITY"
fi

# Fetch stack outputs as JSON
echo "DEBUG: Fetching CloudFormation stack outputs..."
OUTPUTS=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --query 'Stacks[0].Outputs' --output json 2>&1)
FETCH_STATUS=$?
echo "DEBUG: CloudFormation fetch status=$FETCH_STATUS"

if [ $FETCH_STATUS -ne 0 ] || [ "$OUTPUTS" == "null" ]; then
    echo "❌ Failed to fetch outputs for stack: $STACK_NAME"
    echo "DEBUG: OUTPUTS=$OUTPUTS"
    echo "   Make sure the stack exists and you have proper AWS credentials"
    exit 1
fi
echo "DEBUG: Successfully fetched CloudFormation outputs"

# Function to extract output value by key
get_output_value() {
    local key="$1"
    local value=$(echo "$OUTPUTS" | jq -r ".[] | select(.OutputKey == \"$key\") | .OutputValue" 2>/dev/null || echo "")
    echo "DEBUG: get_output_value($key) = $value" >&2
    echo "$value"
}

# Check if jq is available
echo "DEBUG: Checking if jq is available..."
if ! command -v jq &> /dev/null; then
    echo "❌ jq is not installed or not in PATH"
    exit 1
fi
echo "DEBUG: jq found at $(which jq)"

# Extract values from CloudFormation outputs
echo "DEBUG: Extracting values from CloudFormation outputs..."
TENANT_BUCKET=$(get_output_value "tenantBucketName")
USER_POOL_ID=$(get_output_value "frontendUserPoolIdFD4AD2AF")
COGNITO_CLIENT_ID=$(get_output_value "frontendCognitoClientIdDC11CB6A")
COGNITO_IDENTITY_POOL_ID=$(get_output_value "frontendCognitoIdentityPoolIdCA2303C9")
ENVIRONMENT=$(get_output_value "frontendenvironment2580AF7C")
DB_SECRET_NAME= "**********"
REEXTRACT_KEY_FUNCTION_NAME=$(get_output_value "reextractKeyLambdaFunctionName")
echo "DEBUG: Finished extracting CloudFormation outputs"

# Determine region from outputs (extract from ARNs/URLs or default)
echo "DEBUG: Determining region..."
REGION="us-east-2"
if [ -n "$USER_POOL_ID" ]; then
    # Extract region from User Pool ID format: us-east-2_CgOvnB0VX
    REGION=$(echo "$USER_POOL_ID" | cut -d'_' -f1)
fi
echo "DEBUG: REGION=$REGION"

BUCKET_NAME="$TENANT_BUCKET"
echo "DEBUG: BUCKET_NAME=$BUCKET_NAME"

# Construct Cognito URLs
echo "DEBUG: Constructing Cognito URLs..."
COGNITO_ISSUER=""
COGNITO_JWKS_URI=""
if [ -n "$USER_POOL_ID" ]; then
    COGNITO_ISSUER="https://cognito-idp.${REGION}.amazonaws.com/${USER_POOL_ID}"
    COGNITO_JWKS_URI="https://cognito-idp.${REGION}.amazonaws.com/${USER_POOL_ID}/.well-known/jwks.json"
fi
echo "DEBUG: COGNITO_ISSUER=$COGNITO_ISSUER"
echo "DEBUG: COGNITO_JWKS_URI=$COGNITO_JWKS_URI"

# Determine ENV_CLASS based on environment name
echo "DEBUG: Determining ENV_CLASS from ENVIRONMENT=$ENVIRONMENT"
ENV_CLASS="dev"
case "$ENVIRONMENT" in
    *prod*) ENV_CLASS="prod" ;;
    *staging*) ENV_CLASS="staging" ;;
    *dev*) ENV_CLASS="dev" ;;
esac
echo "DEBUG: ENV_CLASS=$ENV_CLASS"

echo "📝 Generating environment files with the following configuration:"
echo "   BUCKET_NAME: $BUCKET_NAME"
echo "   ENV_NAME: $ENVIRONMENT"
echo "   ENV_CLASS: $ENV_CLASS"
echo "   AWS_REGION: $REGION"
echo "   DATABASE_ADMIN_SECRET_NAME: "**********"
echo "   COGNITO_ISSUER: $COGNITO_ISSUER"
echo "   USER_POOL_ID: $USER_POOL_ID"
echo "   COGNITO_CLIENT_ID: $COGNITO_CLIENT_ID"
echo "   REEXTRACT_KEY_FUNCTION_NAME: $REEXTRACT_KEY_FUNCTION_NAME"
echo "   ALGOLIA_SECRETS_PATH: "**********"

echo ""
echo "🔧 Setting up API environment..."
echo "DEBUG: Writing API .env file to $API_ENV_FILE"

# Check if directory exists
API_DIR=$(dirname "$API_ENV_FILE")
echo "DEBUG: Checking if directory exists: $API_DIR"
if [ ! -d "$API_DIR" ]; then
    echo "DEBUG: Directory does not exist, creating: $API_DIR"
    mkdir -p "$API_DIR"
fi
echo "DEBUG: Directory exists, writing file..."

# Generate API .env file
cat > "$API_ENV_FILE" << EOF
# Generated from CloudFormation stack: $STACK_NAME
# Generated on: $(date)

# S3 Bucket Configuration
BUCKET_NAME=$BUCKET_NAME
BACKUP_BUCKET_NAME=$BACKUP_BUCKET

# Environment Configuration
ENV_CLASS=$ENV_CLASS
ENV_NAME=$ENVIRONMENT
AWS_REGION=$REGION

# Database Configuration
DATABASE_ADMIN_SECRET_NAME= "**********"

# Cognito Configuration
COGNITO_ISSUER=$COGNITO_ISSUER
COGNITO_JWKS_URI=$COGNITO_JWKS_URI
COGNITO_USER_POOL_ID=$USER_POOL_ID

# Lambda Function Configuration
REEXTRACT_KEY_FUNCTION_NAME=$REEXTRACT_KEY_FUNCTION_NAME

# Optional Configuration
QUERY_LOGGING=false
PORT=3000
EOF
if [ $? -eq 0 ]; then
    echo "DEBUG: API .env file written successfully"
else
    echo "DEBUG: FAILED to write API .env file"
    exit 1
fi

echo "🔍 Setting up Algolia Sync Lambda environment..."
echo "DEBUG: Writing Algolia Sync .env file to $ALGOLIA_SYNC_ENV_FILE"

# Check if directory exists
ALGOLIA_DIR=$(dirname "$ALGOLIA_SYNC_ENV_FILE")
echo "DEBUG: Checking if directory exists: $ALGOLIA_DIR"
if [ ! -d "$ALGOLIA_DIR" ]; then
    echo "DEBUG: Directory does not exist, creating: $ALGOLIA_DIR"
    mkdir -p "$ALGOLIA_DIR"
fi
echo "DEBUG: Directory exists, writing file..."

# Generate Algolia Sync Lambda .env file
cat > "$ALGOLIA_SYNC_ENV_FILE" << EOF
# Generated from CloudFormation stack: $STACK_NAME
# Generated on: $(date)

# Environment Configuration
ENV_NAME=$ENVIRONMENT
AWS_REGION=$REGION

# Database Configuration
DATABASE_ADMIN_SECRET_NAME= "**********"

# Algolia Configuration (from AWS Secrets Manager)
ALGOLIA_SECRET_ARN=arn: "**********":secretsmanager:us-east-2:637423645815:secret:algolia-dwThc8
ALGOLIA_SECRETS_PATH= "**********"

# Optional: Sentry Configuration (disabled for local dev)
# SENTRY_DSN=
# SENTRY_RELEASE=

# Optional: Direct Algolia credentials for faster local development
# Uncomment and set these to skip Secrets Manager calls during development
# ALGOLIA_APPLICATION_ID=your_algolia_app_id
# ALGOLIA_API_KEY=your_algolia_admin_api_key
EOF
if [ $? -eq 0 ]; then
    echo "DEBUG: Algolia Sync .env file written successfully"
else
    echo "DEBUG: FAILED to write Algolia Sync .env file"
    exit 1
fi

echo "🌐 Setting up frontend environment..."
echo "DEBUG: Writing Frontend .env.local file to $FRONTEND_ENV_FILE"

# Check if directory exists
FRONTEND_DIR=$(dirname "$FRONTEND_ENV_FILE")
echo "DEBUG: Checking if directory exists: $FRONTEND_DIR"
if [ ! -d "$FRONTEND_DIR" ]; then
    echo "DEBUG: Directory does not exist, creating: $FRONTEND_DIR"
    mkdir -p "$FRONTEND_DIR"
fi
echo "DEBUG: Directory exists, writing file..."

# Generate frontend .env.local file
cat > "$FRONTEND_ENV_FILE" << EOF
# Frontend Environment Variables for Local Development
# Generated from CloudFormation stack: $STACK_NAME
# Generated on: $(date)

# Local API Configuration - Point tRPC client to local API
VITE_AWS_API_GATEWAY_URL=http://localhost:3000

# AWS Region
VITE_AWS_REGION=$REGION

# Development Environment
VITE_APP_ENVIRONMENT=dev
VITE_APP_STACK_NAME=$1

# Algolia Configuration
VITE_ALGOLIA_API_KEY=8a75ef1d8723ae5c76aaa65d83feeb58

# Cognito Configuration
$(if [ -n "$USER_POOL_ID" ]; then
    echo "VITE_AWS_COGNITO_USER_POOL_ID=$USER_POOL_ID"
else
    echo "# VITE_AWS_COGNITO_USER_POOL_ID=your-user-pool-id-here"
fi)
$(if [ -n "$COGNITO_CLIENT_ID" ]; then
    echo "VITE_AWS_COGNITO_CLIENT_ID=$COGNITO_CLIENT_ID"
else
    echo "# VITE_AWS_COGNITO_CLIENT_ID=your-client-id-here"
fi)
$(if [ -n "$COGNITO_IDENTITY_POOL_ID" ]; then
    echo "VITE_AWS_COGNITO_IDENTITY_POOL_ID=$COGNITO_IDENTITY_POOL_ID"
else
    echo "# VITE_AWS_COGNITO_IDENTITY_POOL_ID=your-identity-pool-id-here"
fi)

# S3 Bucket Configuration
$(if [ -n "$BUCKET_NAME" ]; then
    echo "VITE_AWS_TENANT_BUCKET_NAME=$BUCKET_NAME"
else
    echo "# VITE_AWS_TENANT_BUCKET_NAME=your-tenant-bucket-name-here"
fi)

# Optional: Sentry Configuration (disabled for local dev)
VITE_SENTRY_DSN=
VITE_SENTRY_RELEASE=
EOF
if [ $? -eq 0 ]; then
    echo "DEBUG: Frontend .env.local file written successfully"
else
    echo "DEBUG: FAILED to write Frontend .env.local file"
    exit 1
fi

# Validate that required variables are set
echo ""
echo "🔍 Validating configuration..."
echo "DEBUG: Starting validation of required variables..."

API_MISSING_VARS=()
[ -z "$BUCKET_NAME" ] && API_MISSING_VARS+=("BUCKET_NAME")
[ -z "$BACKUP_BUCKET" ] && API_MISSING_VARS+=("BACKUP_BUCKET_NAME")
[ -z "$ENVIRONMENT" ] && API_MISSING_VARS+=("ENV_NAME")
[ -z "$DB_SECRET_NAME" ] && API_MISSING_VARS+= "**********"
[ -z "$COGNITO_ISSUER" ] && API_MISSING_VARS+=("COGNITO_ISSUER")
[ -z "$COGNITO_JWKS_URI" ] && API_MISSING_VARS+=("COGNITO_JWKS_URI")
[ -z "$USER_POOL_ID" ] && API_MISSING_VARS+=("COGNITO_USER_POOL_ID")
[ -z "$REEXTRACT_KEY_FUNCTION_NAME" ] && API_MISSING_VARS+=("REEXTRACT_KEY_FUNCTION_NAME")

FRONTEND_MISSING_VARS=()
[ -z "$USER_POOL_ID" ] && FRONTEND_MISSING_VARS+=("USER_POOL_ID")
[ -z "$COGNITO_CLIENT_ID" ] && FRONTEND_MISSING_VARS+=("COGNITO_CLIENT_ID")
[ -z "$COGNITO_IDENTITY_POOL_ID" ] && FRONTEND_MISSING_VARS+=("COGNITO_IDENTITY_POOL_ID")

if [ ${#API_MISSING_VARS[@]} -gt 0 ]; then
    echo "⚠️  API: Some required environment variables could not be determined:"
    printf '   - %s\n' "${API_MISSING_VARS[@]}"
    echo "   Please check your CloudFormation stack outputs and update $API_ENV_FILE manually if needed"
else
    echo "✅ API: Successfully generated $API_ENV_FILE"
fi

if [ ${#FRONTEND_MISSING_VARS[@]} -gt 0 ]; then
    echo "⚠️  Frontend: Some required environment variables could not be determined:"
    printf '   - %s\n' "${FRONTEND_MISSING_VARS[@]}"
    echo "   Please check your CloudFormation stack outputs and update $FRONTEND_ENV_FILE manually if needed"
else
    echo "✅ Frontend: Successfully generated $FRONTEND_ENV_FILE"
fi

ALGOLIA_SYNC_MISSING_VARS=()
[ -z "$ENVIRONMENT" ] && ALGOLIA_SYNC_MISSING_VARS+=("ENV_NAME")
[ -z "$REGION" ] && ALGOLIA_SYNC_MISSING_VARS+=("AWS_REGION")
[ -z "$DB_SECRET_NAME" ] && ALGOLIA_SYNC_MISSING_VARS+= "**********"

if [ ${#ALGOLIA_SYNC_MISSING_VARS[@]} -gt 0 ]; then
    echo "⚠️  Algolia Sync Lambda: Some required environment variables could not be determined:"
    printf '   - %s\n' "${ALGOLIA_SYNC_MISSING_VARS[@]}"
    echo "   Please check your CloudFormation stack outputs and update $ALGOLIA_SYNC_ENV_FILE manually if needed"
else
    echo "✅ Algolia Sync Lambda: Successfully generated $ALGOLIA_SYNC_ENV_FILE"
fi

echo ""
echo "🚀 To start local development:"
echo "   1. Start API: cd src/assets/api && npm run dev"
echo "   2. Start Frontend: cd src/assets/frontend && npm run dev"
echo "   3. Run Algolia Sync (one-time): cd src/assets/lambda/algolia-sync && npm run sync"
echo "   4. Or watch Algolia Sync (continuous): cd src/assets/lambda/algolia-sync && npm run dev"
echo ""
echo "📁 Current directory: $(pwd)"
echo ""
echo "🔍 Configuration:"
echo "   ✓ API configured with real AWS database and services"
echo "   ✓ Frontend configured to use local API (localhost:3000)"
echo "   ✓ Frontend uses real AWS services for auth and storage"
echo "   ✓ Algolia Sync Lambda configured with real AWS services"
echo ""
if [ -n "$USER_POOL_ID" ]; then
    echo "👥 AWS Cognito Console:"
    echo "   https://$REGION.console.aws.amazon.com/cognito/v2/idp/user-pools/$USER_POOL_ID/users?region=$REGION"
    echo ""
fi
echo "💡 All environments are now configured for local development!"
echo "DEBUG: Script completed successfully"
    echo ""
fi
echo "💡 All environments are now configured for local development!"
echo "DEBUG: Script completed successfully"