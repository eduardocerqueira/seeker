#date: 2025-12-18T17:02:00Z
#url: https://api.github.com/gists/8a06b66b769030dd91fa42b24a9bd3cf
#owner: https://api.github.com/users/chaws

FROM debian:latest
                                                                                                                                                                                             
# Install dependencies                                                                                                                                                                       
RUN apt-get update && apt-get install -y \                                                                                                                                                   
    curl \                                                                                                                                                                                   
    git \                                                                                                                                                                                    
    ca-certificates \                                                                                                                                                                        
    gnupg \                                                                                                                                                                                  
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (required for Claude Code)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code globally
RUN npm install -g @anthropic-ai/claude-code

# Create workspace directory
RUN mkdir -p /workspace

# Set working directory
WORKDIR /workspace

# Script to create user matching host UID/GID and run Claude
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]