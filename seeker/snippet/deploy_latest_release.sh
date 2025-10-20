#date: 2025-10-20T17:03:51Z
#url: https://api.github.com/gists/b1be9960542b7f75b15760aaf58c9496
#owner: https://api.github.com/users/praveen-indyzai

#!/usr/bin/env bash
set -e  # Exit immediately if a command fails

# -----------------------------
# CONFIGURATION
# -----------------------------
GITHUB_OWNER="praveen-indyzai"
REPO_NAME="pos-api-svc"
GITHUB_TOKEN="${GITHUB_TOKEN: "**********"
ASSET_NAME="release.tar.gz"
RELEASE_DIR="/var/www/pos-api-svc/release"
TMP_DIR="/tmp/deploy_release"
NODE_ENV="production"
PM2_APP_NAME="pos-api-svc"   # PM2 process name

# Relative path to TypeORM data source file (inside release)
DATASOURCE_FILE="./dist/config/database.js"
MIGRATIONS_DIR="./dist/migrations"

# -----------------------------
# STEP 1: Fetch Latest Release Info
# -----------------------------
echo "📦 Fetching latest release info from GitHub..."
LATEST_RELEASE_API="https://api.github.com/repos/${GITHUB_OWNER}/${REPO_NAME}/releases/latest"

ASSET_URL=$(curl -s -H "Authorization: "**********"
  jq -r "if (.assets | type) == \"array\" then (.assets[] | select(.name==\"${ASSET_NAME}\") | .url) else empty end")


if [ -z "$ASSET_URL" ]; then
    echo "❌ No asset named ${ASSET_NAME} found or release data invalid."
    echo "Check: "**********"
    exit 1
fi

# -----------------------------
# STEP 2: Download Asset
# -----------------------------
mkdir -p "$TMP_DIR"
cd "$TMP_DIR"

echo "⬇️  Downloading latest release asset..."
curl -L -H "Authorization: "**********"
     -H "Accept: application/octet-stream" \
     "$ASSET_URL" \
     -o "$ASSET_NAME"

# -----------------------------
# STEP 3: Extract Files
# -----------------------------
echo "📂 Extracting files..."
tar -xzf "$ASSET_NAME"

EXTRACTED_DIR=$(tar -tzf "$ASSET_NAME" | head -1 | cut -f1 -d"/")
if [ -z "$EXTRACTED_DIR" ]; then
    echo "❌ Could not determine extracted folder."
    exit 1
fi

# -----------------------------
# STEP 4: Deploy to Release Directory
# -----------------------------
echo "🚚 Deploying to $RELEASE_DIR..."
mkdir -p "$RELEASE_DIR"
rsync -a --delete "$EXTRACTED_DIR"/ "$RELEASE_DIR"/

cd "$RELEASE_DIR"

# -----------------------------
# STEP 5: Install Dependencies
# -----------------------------
echo "📦 Installing npm dependencies..."
npm ci --only=production

# -----------------------------
# STEP 6: Auto-Generate Migrations (if entity schema changed)
# -----------------------------
# echo "🧩 Checking for schema changes and generating migration..."
# npx typeorm migration:generate ./src/migrations/AutoMigration -d ./src/data-source.ts || true

# -----------------------------
# STEP 7: Build Project (for TS projects)
# -----------------------------
# if [ -f tsconfig.json ]; then
#   echo "🛠️  Building project..."
#   npm run build
# fi

# -----------------------------
# STEP 8: Run TypeORM Migrations
# -----------------------------
echo "Running migrations..."
npm run typeorm migration:run -- -d "$DATASOURCE_FILE"

# -----------------------------
# STEP 9: Restart PM2 App
# -----------------------------
echo "♻️  Restarting PM2 app: $PM2_APP_NAME..."

if pm2 list | grep -q "$PM2_APP_NAME"; then
    pm2 reload "$PM2_APP_NAME" --update-env
else
    echo "⚠️  PM2 app not found."
    # echo "⚠️  PM2 app not found. Starting new instance..."
    # pm2 start dist/main.js --name "$PM2_APP_NAME"
    # pm2 save
fi

# -----------------------------
# STEP 10: Cleanup
# -----------------------------
echo "🧹 Cleaning up temporary files..."
rm -rf "$TMP_DIR"

echo "🎉 Deployment complete with auto-migrations!"ho "🧹 Cleaning up temporary files..."
rm -rf "$TMP_DIR"

echo "🎉 Deployment complete with auto-migrations!"