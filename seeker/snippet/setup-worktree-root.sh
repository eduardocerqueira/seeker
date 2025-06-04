#date: 2025-06-04T16:56:25Z
#url: https://api.github.com/gists/b1717507a3cc914086e2a40316736d80
#owner: https://api.github.com/users/jamesaphoenix

#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if a worktree path is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}❌ Please provide a worktree path${NC}"
    echo -e "Usage: $0 <worktree-path>"
    echo -e "Example: $0 worktrees/feature-auth"
    exit 1
fi

WORKTREE_PATH=$1

# Check if worktree exists
if [ ! -d "$WORKTREE_PATH" ]; then
    echo -e "${RED}❌ Worktree not found at: $WORKTREE_PATH${NC}"
    echo -e "Create it first with: ./scripts/manage-worktree.sh create <branch-name>"
    exit 1
fi

# Get worktree name from path
WORKTREE_NAME=$(basename "$WORKTREE_PATH")

echo -e "${GREEN}🚀 Setting up worktree with schema isolation${NC}"
echo -e "${YELLOW}📋 Worktree: $WORKTREE_NAME${NC}"
echo -e "${YELLOW}📋 Path: $WORKTREE_PATH${NC}"

# Check if Supabase is running
if ! curl -s http://localhost:54321/health > /dev/null 2>&1; then
    echo -e "${RED}❌ Supabase local stack is not running. Start it with: pnpm db:start${NC}"
    exit 1
fi

# Generate schema name
WORKTREE_SCHEMA="wt_$(echo $WORKTREE_NAME | tr '-' '_' | tr '[:upper:]' '[:lower:]')"
echo -e "${YELLOW}📋 Schema: $WORKTREE_SCHEMA${NC}"

# Create schema in database with extensions
echo -e "${GREEN}📦 Creating database schema: $WORKTREE_SCHEMA${NC}"
psql postgresql://postgres:postgres@localhost:54322/postgres << EOF
-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS $WORKTREE_SCHEMA;

-- Grant usage on schema
GRANT USAGE ON SCHEMA $WORKTREE_SCHEMA TO postgres, anon, authenticated, service_role;
GRANT ALL ON SCHEMA $WORKTREE_SCHEMA TO postgres;
GRANT CREATE ON SCHEMA $WORKTREE_SCHEMA TO postgres;

-- Important: The uuid-ossp extension is already installed in the extensions schema by Supabase
-- We need to reference it properly in our search_path or create wrapper functions

-- Create wrapper function for uuid_generate_v4() in our schema
CREATE OR REPLACE FUNCTION $WORKTREE_SCHEMA.uuid_generate_v4()
RETURNS uuid
LANGUAGE sql
VOLATILE
AS \$\$
SELECT extensions.uuid_generate_v4();
\$\$;

-- Grant execute permissions
GRANT EXECUTE ON FUNCTION $WORKTREE_SCHEMA.uuid_generate_v4() TO postgres, anon, authenticated, service_role;

-- Also create gen_random_uuid wrapper (alternative UUID function)
CREATE OR REPLACE FUNCTION $WORKTREE_SCHEMA.gen_random_uuid()
RETURNS uuid
LANGUAGE sql
VOLATILE
AS \$\$
SELECT gen_random_uuid();
\$\$;

GRANT EXECUTE ON FUNCTION $WORKTREE_SCHEMA.gen_random_uuid() TO postgres, anon, authenticated, service_role;

-- Create wrapper functions for pgcrypto functions (gen_salt, crypt)
CREATE OR REPLACE FUNCTION $WORKTREE_SCHEMA.gen_salt(text)
RETURNS text
LANGUAGE sql
STABLE STRICT
AS \$\$
SELECT extensions.gen_salt(\$1);
\$\$;

CREATE OR REPLACE FUNCTION $WORKTREE_SCHEMA.gen_salt(text, integer)
RETURNS text
LANGUAGE sql
STABLE STRICT
AS \$\$
SELECT extensions.gen_salt(\$1, \$2);
\$\$;

CREATE OR REPLACE FUNCTION $WORKTREE_SCHEMA.crypt(text, text)
RETURNS text
LANGUAGE sql
STABLE STRICT
AS \$\$
SELECT extensions.crypt(\$1, \$2);
\$\$;

GRANT EXECUTE ON FUNCTION $WORKTREE_SCHEMA.gen_salt(text) TO postgres, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION $WORKTREE_SCHEMA.gen_salt(text, integer) TO postgres, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION $WORKTREE_SCHEMA.crypt(text, text) TO postgres, anon, authenticated, service_role;
EOF

# Get Supabase keys
SUPABASE_ANON_KEY=$(cat packages/supabase/supabase/.temp/gotrue/anon.key 2>/dev/null || echo "")
SUPABASE_SERVICE_ROLE_KEY=$(cat packages/supabase/supabase/.temp/gotrue/service.key 2>/dev/null || echo "")

if [ -z "$SUPABASE_ANON_KEY" ] || [ -z "$SUPABASE_SERVICE_ROLE_KEY" ]; then
    # Try to get from supabase status
    echo -e "${YELLOW}⚠️  Reading keys from supabase status${NC}"
    SUPABASE_ANON_KEY=$(cd packages/supabase && supabase status 2>/dev/null | grep "anon key" | cut -d: -f2 | xargs || echo "")
    SUPABASE_SERVICE_ROLE_KEY=$(cd packages/supabase && supabase status 2>/dev/null | grep "service_role key" | cut -d: -f2 | xargs || echo "")
fi

if [ -z "$SUPABASE_ANON_KEY" ] || [ -z "$SUPABASE_SERVICE_ROLE_KEY" ]; then
    echo -e "${YELLOW}⚠️  Could not read Supabase keys - update .env.worktree manually${NC}"
    SUPABASE_ANON_KEY="your-anon-key"
    SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"
fi

# Create .env.worktree file in the worktree
echo -e "${GREEN}📝 Creating .env.worktree configuration${NC}"
cat > "$WORKTREE_PATH/.env.worktree" << EOF
# Worktree Configuration
WORKTREE_NAME=$WORKTREE_NAME
WORKTREE_SCHEMA=$WORKTREE_SCHEMA
DATABASE_SCHEMA=$WORKTREE_SCHEMA
NEXT_PUBLIC_DATABASE_SCHEMA=$WORKTREE_SCHEMA

# Database URLs with schema search path
DATABASE_URL=postgresql://postgres:postgres@localhost:54322/postgres?options=-c%20search_path%3D$WORKTREE_SCHEMA,public
DIRECT_URL=postgresql://postgres:postgres@localhost:54322/postgres?options=-c%20search_path%3D$WORKTREE_SCHEMA,public

# Supabase Configuration
NEXT_PUBLIC_SUPABASE_URL=http://localhost:54321
NEXT_PUBLIC_SUPABASE_ANON_KEY=$SUPABASE_ANON_KEY
SUPABASE_SERVICE_ROLE_KEY=$SUPABASE_SERVICE_ROLE_KEY
SUPABASE_DB_URL=postgresql://postgres:postgres@localhost:54322/postgres?options=-c%20search_path%3D$WORKTREE_SCHEMA,public

# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8080
EOF

# Create docker-compose for this worktree
echo -e "${GREEN}🐳 Creating docker-compose configuration${NC}"

# Generate unique ports for this worktree
# Use a hash of the worktree name to generate consistent ports
HASH=$(echo -n "$WORKTREE_NAME" | md5sum | cut -c1-4)
PORT_OFFSET=$(printf "%d" "0x$HASH")
# Ensure port offset is in a reasonable range (1000-9000)
PORT_OFFSET=$((($PORT_OFFSET % 8000) + 1000))

# Calculate ports
WEB_PORT=$((3000 + $PORT_OFFSET))
API_PORT=$((8080 + $PORT_OFFSET))

# Get the absolute root directory
ROOT_DIR="$(cd "$(dirname "$0")/../" && pwd)"

# Create a standalone docker-compose file for the worktree
cat > "$WORKTREE_PATH/docker-compose.worktree.yml" << EOF
# Generated docker-compose for worktree: $WORKTREE_NAME
# Standalone configuration to avoid port conflicts

services:
  # API service - development with hot reload
  api-dev-$WORKTREE_NAME:
    build:
      context: $ROOT_DIR
      dockerfile: apps/api/Dockerfile
      target: dev
    container_name: octospark-api-$WORKTREE_NAME
    ports:
      - "$API_PORT:8080"
    env_file:
      - $ROOT_DIR/apps/api/.env
    environment:
      - NODE_ENV=development
      - PORT=8080
      - DATABASE_SCHEMA=$WORKTREE_SCHEMA
      - DATABASE_URL=postgresql://postgres:postgres@host.docker.internal:54322/postgres?options=-c%20search_path%3D$WORKTREE_SCHEMA,public
      - SUPABASE_DB_URL=postgresql://postgres:postgres@host.docker.internal:54322/postgres?options=-c%20search_path%3D$WORKTREE_SCHEMA,public
      - NEXT_PUBLIC_SUPABASE_URL=http://localhost:54321
      - NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0
      - IS_DOCKER=true
      - REDIS_URL=redis://redis-$WORKTREE_NAME:6379
    volumes:
      - $ROOT_DIR/apps/api/src:/app/apps/api/src
      - $ROOT_DIR/packages:/app/packages
      - /app/node_modules
      - /app/apps/api/node_modules
    networks:
      - supabase_network_cursordevkit_octospark
    depends_on:
      - redis-$WORKTREE_NAME
    extra_hosts:
      - "host.docker.internal:host-gateway"
    restart: unless-stopped
    profiles:
      - api-dev
      - dev
      - full-stack

  # Web service - Next.js with hot reload
  web-dev-$WORKTREE_NAME:
    build:
      context: $ROOT_DIR
      dockerfile: apps/web/Dockerfile
    container_name: octospark-web-$WORKTREE_NAME
    ports:
      - "$WEB_PORT:3000"
    env_file:
      - $ROOT_DIR/apps/web/.env.local
    environment:
      - NODE_ENV=development
      - PORT=3000
      - DATABASE_SCHEMA=$WORKTREE_SCHEMA
      - NEXT_PUBLIC_DATABASE_SCHEMA=$WORKTREE_SCHEMA
      - NEXT_PUBLIC_SUPABASE_URL=http://localhost:54321
      - NEXT_PUBLIC_API_URL=http://localhost:$API_PORT
      - IS_DOCKER=true
      - WATCHPACK_POLLING=true
      - REDIS_URL=redis://redis-$WORKTREE_NAME:6379
    volumes:
      - $ROOT_DIR/apps/web/app:/app/apps/web/app
      - $ROOT_DIR/apps/web/components:/app/apps/web/components
      - $ROOT_DIR/apps/web/features:/app/apps/web/features
      - $ROOT_DIR/apps/web/hooks:/app/apps/web/hooks
      - $ROOT_DIR/apps/web/lib:/app/apps/web/lib
      - $ROOT_DIR/apps/web/public:/app/apps/web/public
      - $ROOT_DIR/apps/web/stores:/app/apps/web/stores
      - $ROOT_DIR/apps/web/types:/app/apps/web/types
      - $ROOT_DIR/apps/web/utils:/app/apps/web/utils
      - $ROOT_DIR/apps/web/integration:/app/apps/web/integration
      - $ROOT_DIR/apps/web/__tests__:/app/apps/web/__tests__
      - $ROOT_DIR/packages:/app/packages
      - /app/node_modules
      - /app/apps/web/node_modules
      - /app/apps/web/.next
    networks:
      - supabase_network_cursordevkit_octospark
    depends_on:
      - api-dev-$WORKTREE_NAME
    extra_hosts:
      - "host.docker.internal:host-gateway"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    profiles:
      - web-dev
      - dev
      - full-stack

  # Redis service
  redis-$WORKTREE_NAME:
    image: redis:alpine
    container_name: redis-$WORKTREE_NAME
    ports:
      - "$((6379 + $PORT_OFFSET)):6379"
    networks:
      - supabase_network_cursordevkit_octospark
    restart: unless-stopped
    profiles:
      - dev
      - api-dev
      - full-stack

networks:
  supabase_network_cursordevkit_octospark:
    external: true
EOF

# Update the .env.worktree with the new ports
cat >> "$WORKTREE_PATH/.env.worktree" << EOF

# Worktree-specific ports
WEB_PORT=$WEB_PORT
API_PORT=$API_PORT
NEXT_PUBLIC_API_URL=http://localhost:$API_PORT
EOF

# Create helper scripts that can be run from root
echo -e "${GREEN}📄 Creating helper scripts${NC}"

# Create scripts directory
mkdir -p "$WORKTREE_PATH/scripts"

# Create init-worktree-schema.sh script
cat > "$WORKTREE_PATH/scripts/init-worktree-schema.sh" << 'EOF'
#!/bin/bash

# Initialize worktree schema with proper extensions
# This script ensures all required extensions are available in the worktree schema

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check if schema name is provided
if [ -z "${1:-}" ]; then
    echo -e "${RED}Error: Schema name required${NC}"
    echo "Usage: $0 <schema_name>"
    exit 1
fi

SCHEMA_NAME="$1"

echo -e "${GREEN}Initializing schema: $SCHEMA_NAME${NC}"

# Create schema and install extensions
psql postgresql://postgres:postgres@localhost:54322/postgres << EOSQL
-- Create schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS $SCHEMA_NAME;

-- Install extensions in the public schema (they're global)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Grant usage on extensions
GRANT USAGE ON SCHEMA public TO postgres, anon, authenticated, service_role;
GRANT USAGE ON SCHEMA $SCHEMA_NAME TO postgres, anon, authenticated, service_role;
GRANT ALL ON SCHEMA $SCHEMA_NAME TO postgres;

-- Create wrapper functions for extensions in the schema
CREATE OR REPLACE FUNCTION $SCHEMA_NAME.uuid_generate_v4()
RETURNS uuid
LANGUAGE sql
VOLATILE
AS \\\$\\\$
SELECT extensions.uuid_generate_v4();
\\\$\\\$;

CREATE OR REPLACE FUNCTION $SCHEMA_NAME.gen_random_uuid()
RETURNS uuid
LANGUAGE sql
VOLATILE
AS \\\$\\\$
SELECT gen_random_uuid();
\\\$\\\$;

-- Grant execute permissions on wrapper functions
GRANT EXECUTE ON FUNCTION $SCHEMA_NAME.uuid_generate_v4() TO postgres, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION $SCHEMA_NAME.gen_random_uuid() TO postgres, anon, authenticated, service_role;

-- Verify extensions are accessible (session-specific search path)
SET search_path TO $SCHEMA_NAME, public;
SELECT extname, extnamespace::regnamespace 
FROM pg_extension 
WHERE extname IN ('uuid-ossp', 'pgcrypto', 'vector');

-- Test uuid generation works
SELECT $SCHEMA_NAME.uuid_generate_v4() as test_uuid;
EOSQL

echo -e "${GREEN}✅ Schema initialized successfully${NC}"
EOF
chmod +x "$WORKTREE_PATH/scripts/init-worktree-schema.sh"

# Create clean-worktree-schema.sh script
cat > "$WORKTREE_PATH/scripts/clean-worktree-schema.sh" << 'EOF'
#!/bin/bash

# Clean and reset worktree schema
# This allows for a fresh migration run

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source environment to get schema name
source "$ROOT_DIR/.env.worktree"

echo -e "${YELLOW}⚠️  This will DROP and recreate schema: $WORKTREE_SCHEMA${NC}"
echo -e "${YELLOW}All data in this schema will be lost!${NC}"
read -p "Are you sure? (y/N) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Aborted${NC}"
    exit 1
fi

echo -e "${GREEN}Cleaning schema: $WORKTREE_SCHEMA${NC}"

# Drop and recreate schema
psql postgresql://postgres:postgres@localhost:54322/postgres << EOSQL
-- Drop schema cascade (removes all objects)
DROP SCHEMA IF EXISTS $WORKTREE_SCHEMA CASCADE;

-- Recreate schema
CREATE SCHEMA $WORKTREE_SCHEMA;

-- Grant permissions
GRANT ALL ON SCHEMA $WORKTREE_SCHEMA TO postgres;
GRANT USAGE ON SCHEMA $WORKTREE_SCHEMA TO anon, authenticated, service_role;
EOSQL

echo -e "${GREEN}✅ Schema cleaned successfully${NC}"
echo -e "${YELLOW}Run './run-migrations.sh' to apply migrations${NC}"
EOF
chmod +x "$WORKTREE_PATH/scripts/clean-worktree-schema.sh"

# Create a run-migrations script
cat > "$WORKTREE_PATH/run-migrations.sh" << 'EOF'
#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# For worktree, the packages are in the worktree directory itself
ROOT_DIR="$SCRIPT_DIR"

# Source environment
source "$SCRIPT_DIR/.env.worktree"

echo "Running migrations for schema: $WORKTREE_SCHEMA"
echo "From root directory: $ROOT_DIR"

# Initialize schema with extensions first
echo "Initializing schema with required extensions..."
"$SCRIPT_DIR/scripts/init-worktree-schema.sh" "$WORKTREE_SCHEMA"

# Run migrations with the schema-specific search path
cd "$ROOT_DIR/packages/supabase"
for migration in supabase/migrations/*.sql; do
    if [ -f "$migration" ]; then
        echo "Running: $(basename $migration)"
        # Use search_path in the connection string and ensure extensions are accessible
        psql "postgresql://postgres:postgres@localhost:54322/postgres?options=-c%20search_path%3D$WORKTREE_SCHEMA,public" -f "$migration"
    fi
done
cd "$SCRIPT_DIR"

echo "✅ Migrations completed"
EOF
chmod +x "$WORKTREE_PATH/run-migrations.sh"

# Create a seed data script
cat > "$WORKTREE_PATH/seed-worktree.sh" << 'EOF'
#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source environment
source "$SCRIPT_DIR/.env.worktree"

echo "Seeding data for schema: $WORKTREE_SCHEMA"

# Use the fixed schema-aware seed generator that doesn't transform emails
if [ -f "$ROOT_DIR/scripts/generate-schema-seed-fixed.sh" ]; then
    cd "$ROOT_DIR"
    DATABASE_SCHEMA=$WORKTREE_SCHEMA DATABASE_URL=$DATABASE_URL ./scripts/generate-schema-seed-fixed.sh
elif [ -f "$ROOT_DIR/scripts/generate-schema-seed.sh" ]; then
    cd "$ROOT_DIR"
    DATABASE_SCHEMA=$WORKTREE_SCHEMA DATABASE_URL=$DATABASE_URL ./scripts/generate-schema-seed.sh
else
    echo "⚠️  Schema-aware seed script not found"
    echo "    Falling back to standard seed files..."
    
    # Fallback: Run seed files with schema context
    cd "$ROOT_DIR"
    for seed_file in packages/supabase/supabase/seeds/*.sql; do
        if [ -f "$seed_file" ]; then
            echo "Running: $(basename $seed_file)"
            psql "$DATABASE_URL" -c "SET search_path TO $WORKTREE_SCHEMA,public,auth,extensions;" -f "$seed_file"
        fi
    done
fi

echo "✅ Seed data loaded"
EOF
chmod +x "$WORKTREE_PATH/seed-worktree.sh"

# Create test runner script
cat > "$WORKTREE_PATH/test-worktree.sh" << 'EOF'
#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source environment
source "$SCRIPT_DIR/.env.worktree"

echo "Running tests for worktree: $WORKTREE_NAME (schema: $WORKTREE_SCHEMA)"

# Export environment variables for tests
export DATABASE_SCHEMA=$WORKTREE_SCHEMA
export DATABASE_URL=$DATABASE_URL

# Run different test suites from root
cd "$ROOT_DIR"
case "${1:-all}" in
    integration)
        echo "Running integration tests..."
        echo "  - Web app integration tests..."
        cd apps/web && pnpm test:integration
        cd "$ROOT_DIR"
        ;;
    e2e)
        echo "Running e2e tests..."
        cd apps/web && pnpm test:e2e
        cd "$ROOT_DIR"
        ;;
    unit)
        echo "Running unit tests..."
        echo "  - Root unit tests..."
        pnpm test
        echo "  - Core services tests..."
        cd packages/core-services && pnpm test
        cd "$ROOT_DIR"
        echo "  - API tests..."
        cd apps/api && pnpm test
        cd "$ROOT_DIR"
        ;;
    web)
        echo "Running web app tests..."
        cd apps/web && pnpm test
        cd "$ROOT_DIR"
        ;;
    api)
        echo "Running API tests..."
        cd apps/api && pnpm test
        cd "$ROOT_DIR"
        ;;
    core)
        echo "Running core services tests..."
        cd packages/core-services && pnpm test
        cd "$ROOT_DIR"
        ;;
    all)
        echo "Running all tests..."
        echo "  - Root tests..."
        pnpm test
        echo "  - Core services tests..."
        cd packages/core-services && pnpm test
        cd "$ROOT_DIR"
        echo "  - API tests..."
        cd apps/api && pnpm test
        cd "$ROOT_DIR"
        echo "  - Web app tests..."
        cd apps/web && pnpm test
        cd "$ROOT_DIR"
        echo "  - Integration tests..."
        cd apps/web && pnpm test:integration
        cd "$ROOT_DIR"
        ;;
    *)
        echo "Usage: $0 [integration|e2e|unit|web|api|core|all]"
        echo "  integration - Run web app integration tests"
        echo "  e2e        - Run end-to-end tests"
        echo "  unit       - Run all unit tests (root, core, api)"
        echo "  web        - Run web app tests only"
        echo "  api        - Run API tests only"
        echo "  core       - Run core services tests only"
        echo "  all        - Run all tests"
        exit 1
        ;;
esac
EOF
chmod +x "$WORKTREE_PATH/test-worktree.sh"

# Create dev runner script
cat > "$WORKTREE_PATH/dev-worktree.sh" << 'EOF'
#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source environment
source "$SCRIPT_DIR/.env.worktree"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Parse command
CMD="${1:-up}"

case "$CMD" in
  up|start)
    echo -e "${GREEN}Starting services for worktree: $WORKTREE_NAME${NC}"
    echo -e "${YELLOW}Web: http://localhost:$WEB_PORT${NC}"
    echo -e "${YELLOW}API: http://localhost:$API_PORT${NC}"
    cd "$SCRIPT_DIR"
    docker-compose -f docker-compose.worktree.yml --profile dev up -d
    # Show container status
    docker ps --filter "name=octospark-.*-$WORKTREE_NAME"
    ;;
  down|stop)
    echo -e "${YELLOW}Stopping services for worktree: $WORKTREE_NAME${NC}"
    cd "$SCRIPT_DIR"
    docker-compose -f docker-compose.worktree.yml down
    ;;
  restart)
    echo -e "${YELLOW}Restarting services for worktree: $WORKTREE_NAME${NC}"
    cd "$SCRIPT_DIR"
    docker-compose -f docker-compose.worktree.yml restart
    ;;
  logs)
    echo -e "${GREEN}Showing logs for worktree: $WORKTREE_NAME${NC}"
    shift
    cd "$SCRIPT_DIR"
    docker-compose -f docker-compose.worktree.yml logs -f "$@"
    ;;
  status)
    echo -e "${GREEN}Services for worktree: $WORKTREE_NAME${NC}"
    echo -e "${YELLOW}Web: http://localhost:$WEB_PORT${NC}"
    echo -e "${YELLOW}API: http://localhost:$API_PORT${NC}"
    docker ps --filter "name=octospark-.*-$WORKTREE_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    ;;
  *)
    echo -e "${RED}Usage: $0 [up|down|restart|logs|status]${NC}"
    echo -e "  up/start  - Start services in background"
    echo -e "  down/stop - Stop and remove services"
    echo -e "  restart   - Restart services"
    echo -e "  logs      - Show logs (optionally specify service)"
    echo -e "  status    - Show service status"
    exit 1
    ;;
esac
EOF
chmod +x "$WORKTREE_PATH/dev-worktree.sh"

# Create a cleanup script
cat > "$WORKTREE_PATH/cleanup-worktree.sh" << 'EOF'
#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Source environment
source "$SCRIPT_DIR/.env.worktree"

echo "Cleaning up worktree: $WORKTREE_NAME (schema: $WORKTREE_SCHEMA)"

# Stop any running containers
cd "$SCRIPT_DIR"
docker-compose -f docker-compose.worktree.yml down 2>/dev/null || true

# Drop the schema (with CASCADE to remove all objects)
read -p "Drop schema $WORKTREE_SCHEMA? This will delete all data! (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    psql postgresql://postgres:postgres@localhost:54322/postgres -c "DROP SCHEMA IF EXISTS $WORKTREE_SCHEMA CASCADE;"
    echo "✅ Schema dropped"
fi

# Remove generated files
rm -f "$SCRIPT_DIR/.env.worktree" "$SCRIPT_DIR/docker-compose.override.yml" \
      "$SCRIPT_DIR/run-migrations.sh" "$SCRIPT_DIR/seed-worktree.sh" \
      "$SCRIPT_DIR/test-worktree.sh" "$SCRIPT_DIR/dev-worktree.sh" \
      "$SCRIPT_DIR/cleanup-worktree.sh"

echo "✅ Cleanup complete"
EOF
chmod +x "$WORKTREE_PATH/cleanup-worktree.sh"

# Copy environment files from main branch if they exist
echo -e "${GREEN}📋 Copying environment files from main branch${NC}"

# Copy API .env file if it exists
if [ -f "apps/api/.env" ]; then
    cp "apps/api/.env" "$WORKTREE_PATH/apps/api/.env"
    echo -e "${GREEN}✓ Copied apps/api/.env${NC}"
else
    echo -e "${YELLOW}⚠ No apps/api/.env found in main branch${NC}"
fi

# Copy Web .env.local file if it exists
if [ -f "apps/web/.env.local" ]; then
    cp "apps/web/.env.local" "$WORKTREE_PATH/apps/web/.env.local"
    # Update the API URL in the web env file to use the worktree-specific port
    if command -v sed >/dev/null 2>&1; then
        sed -i.bak "s|NEXT_PUBLIC_API_URL=.*|NEXT_PUBLIC_API_URL=http://localhost:$API_PORT|g" "$WORKTREE_PATH/apps/web/.env.local"
        rm -f "$WORKTREE_PATH/apps/web/.env.local.bak"
        echo -e "${GREEN}✓ Copied and updated apps/web/.env.local${NC}"
    else
        echo -e "${GREEN}✓ Copied apps/web/.env.local (manual port update needed)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ No apps/web/.env.local found in main branch${NC}"
fi

# Add gitignore entries if not already present
if ! grep -q ".env.worktree" "$WORKTREE_PATH/.gitignore" 2>/dev/null; then
    echo -e "\n# Worktree specific files" >> "$WORKTREE_PATH/.gitignore"
    echo ".env.worktree" >> "$WORKTREE_PATH/.gitignore"
    echo "docker-compose.worktree.yml" >> "$WORKTREE_PATH/.gitignore"
    echo "run-migrations.sh" >> "$WORKTREE_PATH/.gitignore"
    echo "seed-worktree.sh" >> "$WORKTREE_PATH/.gitignore"
    echo "test-worktree.sh" >> "$WORKTREE_PATH/.gitignore"
    echo "dev-worktree.sh" >> "$WORKTREE_PATH/.gitignore"
    echo "cleanup-worktree.sh" >> "$WORKTREE_PATH/.gitignore"
    echo -e "${GREEN}📝 Added worktree files to .gitignore${NC}"
fi

echo -e "${GREEN}✅ Worktree setup complete!${NC}"
echo ""
echo -e "Next steps (run from the root directory):"
echo -e "1. Run migrations: ${YELLOW}$WORKTREE_PATH/run-migrations.sh${NC}"
echo -e "2. Seed data: ${YELLOW}$WORKTREE_PATH/seed-worktree.sh${NC}"
echo -e "3. Start development:"
echo -e "   - Docker: ${YELLOW}$WORKTREE_PATH/dev-worktree.sh${NC}"
echo -e "   - Without Docker: ${YELLOW}source $WORKTREE_PATH/.env.worktree && pnpm dev${NC}"
echo -e "4. Run tests: ${YELLOW}$WORKTREE_PATH/test-worktree.sh [integration|e2e|unit|all]${NC}"
echo -e "5. When done: ${YELLOW}$WORKTREE_PATH/cleanup-worktree.sh${NC}"
echo ""
echo -e "Schema: ${GREEN}$WORKTREE_SCHEMA${NC}"
echo -e "Database: ${GREEN}postgresql://localhost:54322${NC}"