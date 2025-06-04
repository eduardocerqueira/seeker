#date: 2025-06-04T16:56:25Z
#url: https://api.github.com/gists/b1717507a3cc914086e2a40316736d80
#owner: https://api.github.com/users/jamesaphoenix

#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get schema from environment or use 'public'
SCHEMA=${DATABASE_SCHEMA:-public}

echo -e "${GREEN}🌱 Generating schema-aware seed file for: $SCHEMA${NC}"
echo -e "${YELLOW}📋 Using shared auth.users (no email transformation)${NC}"

# For worktree schemas, we handle auth.users differently - we share them
if [ "$SCHEMA" != "public" ]; then
    echo -e "${YELLOW}🔄 Ensuring shared auth.users exist...${NC}"
    
    USER_COUNT=$(psql "${DATABASE_URL}" -t -c "SELECT COUNT(*) FROM auth.users;" | xargs)
    
    if [ "$USER_COUNT" -eq "0" ]; then
        echo -e "${YELLOW}⚠️  No users in auth.users. Running public seed first...${NC}"
        
        # Run the original users seed file in public schema to populate auth.users
        for seed_file in packages/supabase/supabase/seeds/*.sql; do
            if [ -f "$seed_file" ]; then
                filename=$(basename "$seed_file")
                if [[ "$filename" == "02_users_and_identities.sql" ]]; then
                    echo -e "${YELLOW}Seeding auth.users from: $filename${NC}"
                    psql "${DATABASE_URL}" -c "SET search_path TO public,auth;" -f "$seed_file"
                fi
            fi
        done
    else
        echo -e "${GREEN}✅ Found $USER_COUNT users in auth.users${NC}"
    fi
fi

# Now seed the schema-specific tables
for seed_file in packages/supabase/supabase/seeds/*.sql; do
    if [ -f "$seed_file" ]; then
        filename=$(basename "$seed_file")        
        echo -e "${YELLOW}Processing: $filename${NC}"
        
        # Skip auth.users seeding for non-public schemas (we're using shared auth.users)
        if [[ "$filename" == "02_users_and_identities.sql" ]] && [ "$SCHEMA" != "public" ]; then
            echo -e "${YELLOW}↪️  Skipping auth.users seeding (using shared users)${NC}"
            
            # Instead, sync the users table in our schema with auth.users
            echo -e "${YELLOW}🔄 Syncing ${SCHEMA}.users with auth.users${NC}"
            psql "${DATABASE_URL}" << EOF
-- Sync users from auth.users to schema-specific users table
INSERT INTO ${SCHEMA}.users (id, email, created_at, updated_at)
SELECT 
    au.id, 
    au.email, 
    au.created_at, 
    au.updated_at 
FROM auth.users au
WHERE NOT EXISTS (
    SELECT 1 FROM ${SCHEMA}.users u WHERE u.id = au.id
)
ON CONFLICT (id) DO UPDATE SET
    email = EXCLUDED.email,
    updated_at = EXCLUDED.updated_at;

-- Log the sync
DO \$\$
DECLARE
    user_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO user_count FROM ${SCHEMA}.users;
    RAISE NOTICE 'Synced % users to %.users', user_count, '${SCHEMA}';
END\$\$;
EOF
            echo -e "${GREEN}✅ User sync completed${NC}"
        else
            # For other seed files, run them with schema context
            # Set search path to prioritize our schema, then public, then auth
            echo -e "${YELLOW}🔧 Running with schema search path: ${SCHEMA},public,auth${NC}"
            psql "${DATABASE_URL}" -c "SET search_path TO ${SCHEMA},public,auth;" -f "$seed_file"
        fi
    fi
done

# Verify the setup
echo -e "${GREEN}📊 Verification:${NC}"
psql "${DATABASE_URL}" << EOF
-- Check user counts
DO \$\$
DECLARE
    auth_user_count INTEGER;
    schema_user_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO auth_user_count FROM auth.users;
    
    -- Only check schema users if not in public schema
    IF '${SCHEMA}' != 'public' THEN
        SELECT COUNT(*) INTO schema_user_count FROM ${SCHEMA}.users;
        RAISE NOTICE 'auth.users count: %', auth_user_count;
        RAISE NOTICE '${SCHEMA}.users count: %', schema_user_count;
        
        IF auth_user_count != schema_user_count THEN
            RAISE WARNING 'User counts do not match! This might indicate a sync issue.';
        END IF;
    ELSE
        RAISE NOTICE 'auth.users count: %', auth_user_count;
        RAISE NOTICE 'Running in public schema - no sync needed';
    END IF;
END\$\$;
EOF

echo -e "${GREEN}✅ Schema-aware seeding complete for: $SCHEMA${NC}"
echo -e "${YELLOW}ℹ️  Note: Using shared auth.users across all schemas (no email transformation)${NC}" 