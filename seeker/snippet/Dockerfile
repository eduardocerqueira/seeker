#date: 2024-08-14T18:12:56Z
#url: https://api.github.com/gists/80c8d32519a4bcb1d59608d0651a8981
#owner: https://api.github.com/users/tuffacton

# Add kubectl to the Docker container
FROM alpine:latest AS kubectl

# Specify the kubectl version to download
ARG KUBECTL_VERSION=v1.28.2

# Install kubectl
RUN apk add --no-cache curl && \
    curl -LO "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/$(uname -m)/kubectl" && \
    chmod +x kubectl && \
    mv kubectl /usr/local/bin/
    
# Copy kubectl into the main image
FROM moby/buildkit:master-rootless AS final-image
COPY --from=kubectl /usr/local/bin/kubectl /usr/local/bin/kubectl