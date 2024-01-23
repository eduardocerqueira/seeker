#date: 2024-01-23T17:04:23Z
#url: https://api.github.com/gists/7d9fecc239cb3b4909f06265b737604c
#owner: https://api.github.com/users/liropa

# Use the official Rust image as a base
FROM rust:1.57

# Set the working directory
WORKDIR /usr/src/volta

# Clone the Volta repository
RUN git clone --branch v1.1.1 https://github.com/volta-cli/volta.git .

# Build Volta
RUN cargo build --release

# Add the Volta bin folder to the PATH
ENV PATH="/usr/src/volta/target/release:${PATH}"

# Setup volta
RUN volta setup

# Add managed packages from Volta to the PATH
ENV PATH="/root/.volta/bin:${PATH}"
