#date: 2025-09-04T16:54:21Z
#url: https://api.github.com/gists/6a157cde2199221cc262fcaccb3dc639
#owner: https://api.github.com/users/brunoalbim

# VARIAVEIS DO SCRIPT
IMAGE_NAME="chatbot-agent"
IMAGE_TAG="latest"
USERNAME="brunoalbim"
IP="IP"



# [1] BUILD DA IMAGEM DO DOCKER

# Script para construir a imagem Docker com a tag "$IMAGE_NAME:$IMAGE_TAG"
docker build -t $IMAGE_NAME:$IMAGE_TAG .
# ou
# [RECOMENDADO] Script para construir a imagem Docker com a tag "$IMAGE_NAME:$IMAGE_TAG" para a plataforma linux/amd64
docker buildx build --platform linux/amd64 -t $IMAGE_NAME:$IMAGE_TAG .
# ou
# Script para construir a imagem Docker com a tag "$USERNAME/$IMAGE_NAME:$IMAGE_TAG" para múltiplas plataformas
docker buildx build --platform linux/amd64,linux/arm64 -t $USERNAME/$IMAGE_NAME:$IMAGE_TAG .



# [2] SALVAR E TRANSFERIR A IMAGEM PARA A VPS

# Salvar a imagem Docker em um arquivo tar
docker save -o $IMAGE_NAME.tar $USERNAME/$IMAGE_NAME:$IMAGE_TAG

# Transferir o arquivo tar para a VPS usando scp
scp $IMAGE_NAME.tar root@$IP:~



# [3] CARREGAR IMAGEM NO DOCKER

# Carregar a imagem da aplicação no Docker
docker load -i $IMAGE_NAME.tar

# Remover o arquivo tar após a importação (opcional)
rm -rf $IMAGE_NAME.tar