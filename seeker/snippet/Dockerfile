#date: 2022-10-10T17:33:52Z
#url: https://api.github.com/gists/d3fcdd10535e166af8caf88409d9e448
#owner: https://api.github.com/users/ibLeDy

# Tags should be as specific as possible
FROM ubuntu:jammy-20220815

# Add metadata to the image
LABEL maintainer="ops@marfeel.com"

# Use diff-friendly syntax for multi-line commands
# Do not install non-necessary packages
# Sort dependencies
# Clean non-necessary files after installing dependencies
RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        curl \
        sl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :

# Do not cache Python dependencies
# Sort dependencies
RUN : \
    && python3 -m pip --no-cache-dir install \
        pre-commit \
        virtualenv \
    && :

# Create non-privileged user
RUN : \
    && groupadd --gid 1001 marfeel \
    && useradd --uid 1001 --gid marfeel --system --create-home --home-dir /home/marfeel marfeel \
    && :

# Do not use an extra step to change ownership of files
COPY --chown=marfeel:marfeel Dockerfile /Dockerfile    

# Set non-privileged user as default
USER marfeel
