#date: 2025-11-06T17:10:48Z
#url: https://api.github.com/gists/abd002fe6b3db1dcedc05d5cafafdf98
#owner: https://api.github.com/users/ghostturnled2000-boop

#!/bin/bash

# Bytebot OS + Ollama API - Full Automated Setup
# This script sets up Bytebot OS with a local free Ollama API
# No cost, no API keys needed - runs completely locally!

set -e

echo "========================================"
echo "Bytebot OS + Ollama Setup"
echo "Free Local AI Stack Automation"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}[1/6] Checking Prerequisites...${NC}"
sleep 1

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker not found. Installing...${NC}"
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo -e "${GREEN}Docker installed!${NC}"
else
    echo -e "${GREEN}Docker found!${NC}"
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose not found. Installing...${NC}"
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo -e "${GREEN}Docker Compose installed!${NC}"
else
    echo -e "${GREEN}Docker Compose found!${NC}"
fi

echo ""
echo -e "${YELLOW}[2/6] Cloning Bytebot OS Repository...${NC}"
sleep 1

if [ ! -d "bytebot" ]; then
    git clone https://github.com/bytebot-ai/bytebot.git
    echo -e "${GREEN}Repository cloned!${NC}"
else
    echo -e "${GREEN}Repository already exists!${NC}"
fi

cd bytebot

echo ""
echo -e "${YELLOW}[3/6] Configuring Ollama (Local Free API)...${NC}"
sleep 1

mkdir -p docker

# Create environment file with Ollama configuration
cat > docker/.env << 'EOF'
# Bytebot OS Configuration
# Using Ollama as free local LLM API
BYTEBOT_LLM_MODEL=ollama/llama2
BYTEBOT_LLM_API_BASE=http://ollama:11434
BYTEBOT_LLM_API_KEY=sk-local
EOF

echo -e "${GREEN}Environment configured for Ollama!${NC}"

echo ""
echo -e "${YELLOW}[4/6] Creating Enhanced Docker Compose File...${NC}"
sleep 1

# Check if docker-compose file exists and backup if needed
if [ -f "docker/docker-compose.yml" ]; then
    cp docker/docker-compose.yml docker/docker-compose.yml.backup
fi

# Create docker-compose with Ollama integration
cat > docker/docker-compose.yml << 'COMPOSE_EOF'
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    container_name: bytebot-ollama
    ports:
      - "11434:11434"
    environment:
      - OLLAMA_HOST=0.0.0.0:11434
    volumes:
      - ollama_data:/root/.ollama
    command: serve
    restart: unless-stopped
    networks:
      - bytebot-network

  ollama-pull:
    image: ollama/ollama:latest
    container_name: ollama-model-puller
    depends_on:
      - ollama
    environment:
      - OLLAMA_HOST=http://ollama:11434
    volumes:
      - ollama_data:/root/.ollama
    command: >
      sh -c 'echo "Pulling Llama 2 model..." &&
             curl -X POST http://ollama:11434/api/pull -d {\"name\":\"llama2\"}  &&
             echo "Model ready!"'
    restart: on-failure
    networks:
      - bytebot-network

  bytebot-desktop:
    image: bytebot-ai/bytebot-desktop:latest
    container_name: bytebot-desktop
    ports:
      - "5900:5900"
    environment:
      - DISPLAY=:0
    restart: unless-stopped
    networks:
      - bytebot-network

  bytebot-agent:
    image: bytebot-ai/bytebot-agent:latest
    container_name: bytebot-agent
    ports:
      - "9991:9991"
    depends_on:
      - ollama
      - bytebot-desktop
    environment:
      - LLM_API_BASE=http://ollama:11434
      - LLM_MODEL=llama2
      - DESKTOP_HOST=bytebot-desktop
    restart: unless-stopped
    networks:
      - bytebot-network

  bytebot-ui:
    image: bytebot-ai/bytebot-ui:latest
    container_name: bytebot-ui
    ports:
      - "9992:3000"
    depends_on:
      - bytebot-agent
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:9991
    restart: unless-stopped
    networks:
      - bytebot-network

  postgres:
    image: postgres:15
    container_name: bytebot-postgres
    environment:
      - POSTGRES_PASSWORD= "**********"
      - POSTGRES_DB=bytebot
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    networks:
      - bytebot-network

volumes:
  ollama_data:
  postgres_data:

networks:
  bytebot-network:
    driver: bridge
COMPOSE_EOF

echo -e "${GREEN}Docker Compose configured!${NC}"

echo ""
echo -e "${YELLOW}[5/6] Starting Bytebot OS + Ollama...${NC}"
sleep 1

echo -e "${YELLOW}This may take a few minutes on first run...${NC}"
docker-compose -f docker/docker-compose.yml up -d

echo -e "${GREEN}Services started!${NC}"

echo ""
echo -e "${YELLOW}[6/6] Waiting for Services to be Ready...${NC}"
sleep 1

# Wait for services to be ready
echo "Waiting for Bytebot UI (max 2 minutes)..."
for i in {1..120}; do
    if curl -s http://localhost:9992 > /dev/null 2>&1; then
        echo -e "${GREEN}Bytebot UI is ready!${NC}"
        break
    fi
    if [ $((i % 10)) -eq 0 ]; then
        echo "Waiting... ($i seconds)"
    fi
    sleep 1
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Bytebot OS + Ollama Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Access your setup:${NC}"
echo -e "  ${GREEN}Dashboard:${NC} http://localhost:9992"
echo -e "  ${GREEN}Agent API:${NC} http://localhost:9991"
echo -e "  ${GREEN}Ollama API:${NC} http://localhost:11434"
echo -e "  ${GREEN}Desktop VNC:${NC} localhost:5900"
echo ""
echo -e "${YELLOW}Useful Commands:${NC}"
echo -e "  View logs:        ${GREEN}docker-compose -f docker/docker-compose.yml logs -f${NC}"
echo -e "  Stop services:    ${GREEN}docker-compose -f docker/docker-compose.yml down${NC}"
echo -e "  Stop + Remove:    ${GREEN}docker-compose -f docker/docker-compose.yml down -v${NC}"
echo ""
echo -e "${YELLOW}Opening Bytebot Dashboard in browser...${NC}"

# Try to open browser automatically
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:9992
elif command -v open &> /dev/null; then
    open http://localhost:9992
else
    echo -e "${YELLOW}Please open http://localhost:9992 in your browser${NC}"
fi

echo ""
echo -e "${GREEN}Setup complete! Bytebot OS is now running with free local Ollama AI.${NC}"
echo -e "${GREEN}All services running without cloud dependencies or API costs!${NC}"
ts!${NC}"
