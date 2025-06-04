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

echo -e "${GREEN}🚀 Setting up worktree with schema isolation${NC}"

# Check if we're in a git worktree or if it's a regular branch checkout
IS_WORKTREE=false
if [ -f .git ]; then
    # This is a worktree (.git is a file pointing to the git directory)
    IS_WORKTREE=true
    WORKTREE_NAME=$(basename $(pwd))
elif [ -d .git ]; then
    # This is the main repository or a regular checkout
    # Check if we're in a worktree directory (under worktrees/)
    if [[ $(pwd) == *"/worktrees/"* ]]; then
        IS_WORKTREE=true
        WORKTREE_NAME=$(basename $(pwd))
    else
        echo -e "${YELLOW}⚠️  Running in main repository. Using branch name for schema.${NC}"
        WORKTREE_NAME=$(git branch --show-current)
        if [ "$WORKTREE_NAME" == "main" ] || [ "$WORKTREE_NAME" == "master" ]; then
            echo -e "${RED}❌ Cannot use schema isolation on main/master branch${NC}"
            echo -e "${YELLOW}   Please create a feature branch or worktree${NC}"
            exit 1
        fi
    fi
else
    echo -e "${RED}❌ Not in a git repository${NC}"
    exit 1
fi

# Check if Supabase is running
if ! curl -s http://localhost:54321/health > /dev/null 2>&1; then
    echo -e "${RED}❌ Supabase local stack is not running. Start it with: pnpm db:start${NC}"
    exit 1
fi

# Generate schema name
WORKTREE_SCHEMA="wt_$(echo $WORKTREE_NAME | tr '-' '_' | tr '[:upper:]' '[:lower:]')"

echo -e "${YELLOW}📋 Branch/Worktree: $WORKTREE_NAME${NC}"
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
EOF

# Get Supabase keys
SUPABASE_ANON_KEY=$(cat packages/supabase/supabase/.temp/gotrue/anon.key 2>/dev/null || echo "")
SUPABASE_SERVICE_ROLE_KEY=$(cat packages/supabase/supabase/.temp/gotrue/service.key 2>/dev/null || echo "")

if [ -z "$SUPABASE_ANON_KEY" ] || [ -z "$SUPABASE_SERVICE_ROLE_KEY" ]; then
    echo -e "${YELLOW}⚠️  Could not read Supabase keys from .temp directory${NC}"
    echo -e "${YELLOW}   Using placeholder keys - update .env.worktree manually${NC}"
    SUPABASE_ANON_KEY="your-anon-key"
    SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"
fi

# Create .env.worktree file
echo -e "${GREEN}📝 Creating .env.worktree configuration${NC}"
cat > .env.worktree << EOF
# Worktree Configuration
WORKTREE_NAME=$WORKTREE_NAME
WORKTREE_SCHEMA=$WORKTREE_SCHEMA
DATABASE_SCHEMA=$WORKTREE_SCHEMA

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

# Create docker-compose override for this worktree (only in worktrees)
if [ "$IS_WORKTREE" = true ]; then
    echo -e "${GREEN}🐳 Creating docker-compose override configuration${NC}"
    cat > docker-compose.override.yml << EOF
# Generated docker-compose override for worktree: $WORKTREE_NAME
# This file is automatically loaded by docker-compose
version: '3.8'

services:
  api-dev:
    environment:
      - DATABASE_SCHEMA=$WORKTREE_SCHEMA
      - DATABASE_URL=postgresql://postgres:postgres@host.docker.internal:54322/postgres?options=-c%20search_path%3D$WORKTREE_SCHEMA,public
      - SUPABASE_DB_URL=postgresql://postgres:postgres@host.docker.internal:54322/postgres?options=-c%20search_path%3D$WORKTREE_SCHEMA,public
    extra_hosts:
      - "host.docker.internal:host-gateway"

  web-dev:
    environment:
      - DATABASE_SCHEMA=$WORKTREE_SCHEMA
    extra_hosts:
      - "host.docker.internal:host-gateway"
EOF
    echo -e "${YELLOW}   Note: docker-compose will automatically use docker-compose.override.yml${NC}"
else
    echo -e "${YELLOW}⚠️  Not in a worktree - set DATABASE_SCHEMA env var manually for docker-compose${NC}"
fi

# Create a migration runner script
echo -e "${GREEN}📄 Creating migration runner script${NC}"
cat > run-migrations.sh << 'EOF'
#!/bin/bash
set -e

source .env.worktree

echo "Running migrations for schema: $WORKTREE_SCHEMA"

# Run migrations with the schema-specific search path
cd packages/supabase
for migration in supabase/migrations/*.sql; do
    if [ -f "$migration" ]; then
        echo "Running: $(basename $migration)"
        psql "$DATABASE_URL" -f "$migration"
    fi
done
cd ../..

echo "✅ Migrations completed"
EOF
chmod +x run-migrations.sh

# Create a seed data script
echo -e "${GREEN}📊 Creating seed data script${NC}"
cat > seed-worktree.sh << 'EOF'
#!/bin/bash
set -e

source .env.worktree

echo "Seeding data for schema: $WORKTREE_SCHEMA"

# Use the schema-aware seed generator
if [ -f "./scripts/generate-schema-seed.sh" ]; then
    ./scripts/generate-schema-seed.sh
else
    echo "⚠️  Schema-aware seed script not found"
    echo "    Falling back to standard seed files..."
    
    # Fallback: Run seed files with schema context
    for seed_file in packages/supabase/supabase/seeds/*.sql; do
        if [ -f "$seed_file" ]; then
            echo "Running: $(basename $seed_file)"
            psql "$DATABASE_URL" -c "SET search_path TO $WORKTREE_SCHEMA;" -f "$seed_file"
        fi
    done
fi

echo "✅ Seed data loaded"
EOF
chmod +x seed-worktree.sh

# Create test runner script
echo -e "${GREEN}🧪 Creating test runner script${NC}"
cat > test-worktree.sh << 'EOF'
#!/bin/bash
set -e

source .env.worktree

echo "Running tests for worktree: $WORKTREE_NAME (schema: $WORKTREE_SCHEMA)"

# Export environment variables for tests
export DATABASE_SCHEMA=$WORKTREE_SCHEMA
export DATABASE_URL=$DATABASE_URL

# Run different test suites
case "${1:-all}" in
    integration)
        echo "Running integration tests..."
        cd apps/web && pnpm test:integration
        ;;
    e2e)
        echo "Running e2e tests..."
        cd apps/web && pnpm test:e2e
        ;;
    unit)
        echo "Running unit tests..."
        pnpm test
        ;;
    all)
        echo "Running all tests..."
        pnpm test
        cd apps/web && pnpm test:integration
        ;;
    *)
        echo "Usage: ./test-worktree.sh [integration|e2e|unit|all]"
        exit 1
        ;;
esac
EOF
chmod +x test-worktree.sh

# Create convenience scripts
echo -e "${GREEN}🔧 Creating convenience scripts${NC}"

# Docker compose runner that uses environment variables
cat > dev-worktree.sh << 'EOF'
#!/bin/bash
source .env.worktree
# docker-compose automatically uses docker-compose.override.yml if it exists
docker-compose --profile dev up "$@"
EOF
chmod +x dev-worktree.sh

# Create a cleanup script for when done with worktree
cat > cleanup-worktree.sh << 'EOF'
#!/bin/bash
set -e
source .env.worktree

echo "Cleaning up worktree: $WORKTREE_NAME (schema: $WORKTREE_SCHEMA)"

# Stop any running containers
docker-compose down 2>/dev/null || true

# Drop the schema (with CASCADE to remove all objects)
read -p "Drop schema $WORKTREE_SCHEMA? This will delete all data! (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    psql postgresql://postgres:postgres@localhost:54322/postgres -c "DROP SCHEMA IF EXISTS $WORKTREE_SCHEMA CASCADE;"
    echo "✅ Schema dropped"
fi

# Remove generated files
rm -f .env.worktree docker-compose.override.yml run-migrations.sh seed-worktree.sh test-worktree.sh dev-worktree.sh cleanup-worktree.sh

echo "✅ Cleanup complete"
EOF
chmod +x cleanup-worktree.sh

echo -e "${GREEN}✅ Worktree setup complete!${NC}"
echo ""
echo -e "Next steps:"
echo -e "1. Run migrations: ${YELLOW}./run-migrations.sh${NC}"
echo -e "2. Seed data: ${YELLOW}./seed-worktree.sh${NC}"
echo -e "3. Start development:"
echo -e "   - Docker: ${YELLOW}./dev-worktree.sh${NC} (or just ${YELLOW}docker-compose --profile dev up${NC})"
echo -e "   - Without Docker: ${YELLOW}source .env.worktree && pnpm dev${NC}"
echo -e "4. Run tests: ${YELLOW}./test-worktree.sh [integration|e2e|unit|all]${NC}"
echo -e "5. When done: ${YELLOW}./cleanup-worktree.sh${NC}"
echo ""
echo -e "Schema: ${GREEN}$WORKTREE_SCHEMA${NC}"
echo -e "Database: ${GREEN}postgresql://localhost:54322${NC}"

# Add gitignore entries if not already present
if ! grep -q ".env.worktree" .gitignore 2>/dev/null; then
    echo -e "\n# Worktree specific files" >> .gitignore
    echo ".env.worktree" >> .gitignore
    echo "docker-compose.override.yml" >> .gitignore
    echo "run-migrations.sh" >> .gitignore
    echo "seed-worktree.sh" >> .gitignore
    echo "test-worktree.sh" >> .gitignore
    echo "dev-worktree.sh" >> .gitignore
    echo "cleanup-worktree.sh" >> .gitignore
    echo -e "${GREEN}📝 Added worktree files to .gitignore${NC}"
fi