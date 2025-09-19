#date: 2025-09-19T17:09:35Z
#url: https://api.github.com/gists/39d6046183c0be18df93b3255e1622ef
#owner: https://api.github.com/users/teamloset

#!/bin/bash
# Fix deployment script to ensure proper configuration

set -e

echo "🔧 Fixing deployment configuration..."

# Navigate to project directory
cd /opt/extractos-bancarios-odoo

# Pull latest changes
echo "📥 Pulling latest changes..."
git pull origin main

# Stop containers
echo "🛑 Stopping containers..."
docker-compose down || true

# Clean old images
echo "🗑️ Cleaning old images..."
docker image prune -f

# Rebuild and start
echo "🏗️ Building and starting containers..."
docker-compose up -d --build

# Wait for startup
echo "⏳ Waiting for startup..."
sleep 20

# Check container status
echo "✅ Container status:"
docker ps | grep extractos

# Check health
echo "🩺 Health check:"
curl -f http://localhost:8501/_stcore/health || echo "❌ Health check failed"

# Verify Nginx configuration
echo "🌐 Testing Nginx proxy..."
curl -I http://localhost:80 || echo "❌ Nginx proxy issue"

echo "🎉 Deployment fix completed!"