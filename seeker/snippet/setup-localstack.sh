#date: 2023-12-22T16:53:44Z
#url: https://api.github.com/gists/8e6b8841ee3ebd4d1e082177feb201be
#owner: https://api.github.com/users/neeraj-banknovo


echo "🐳 Docker container setting up..." 
docker compose -f localstack-compose.yaml up -d  

echo "⌛️ Hold on.. waiting for localstack to be ready"

until $(curl --output /dev/null --silent --head --fail http://localhost:4566/health); do
    printf '....'
    sleep 5
done

echo "🎉 Localstack is ready!"
echo "🛠️ Setting localstack url to environment variable..."

export AWS_ENDPOINT_URL=http://localhost:4566

echo "🛠️ Setting up the aws configs.."
aws configure set aws_access_key_id testing
aws configure set aws_secret_access_key testing
aws configure set default.region us-east-1

echo "🚀 Everything is all set!"
