#date: 2025-06-05T16:48:12Z
#url: https://api.github.com/gists/52b9b6e0c04a709d572d35d696ae12be
#owner: https://api.github.com/users/ZachS

BASE_DIR="/home/forge/site.com"
## APPLICATION WEB DIRECTORY MUST BE SET TO "/current/public"

echo "Starting deployment..."

# Define paths using Forge variables
RELEASES_DIR="$BASE_DIR/releases"
CURRENT_DIR="$BASE_DIR/current"
STORAGE_DIR="$BASE_DIR/storage"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
RELEASE_DIR="$RELEASES_DIR/$TIMESTAMP"
APP_DIR="$BASE_DIR/app"

# Navigate to base directory
cd $BASE_DIR

# Check if the app directory exists
if [ -d "$APP_DIR" ]; then
    echo "First Run: Found existing app directory. Moving to new release directory..."
    mkdir -p $RELEASE_DIR
    shopt -s dotglob nullglob
    for item in *; do
        if [ "$item" != "releases" ] && [ "$item" != "current" ] && [ "$item" != "storage" ] && [ "$item" != ".env" ]; then
            mv "$item" "$RELEASE_DIR/"
        fi
    done
    shopt -u dotglob nullglob
else
    LATEST_RELEASE=$(ls -td $RELEASES_DIR/* | head -n 1)
    if [ -d "$LATEST_RELEASE" ]; then
        echo "Copying previous release directory to new release directory..."
        cp -a $LATEST_RELEASE/. $RELEASE_DIR/
    fi
fi

cd $RELEASE_DIR/
git reset --hard origin/$FORGE_SITE_BRANCH
git pull origin -f $FORGE_SITE_BRANCH
git clean -f

# Link storage directory
echo "Linking storage directory..."
rm -rf storage
ln -s $STORAGE_DIR storage

# Link .env file
echo "Linking .env file..."
rm -f .env
ln -sf $BASE_DIR/.env .env

echo "Installing Composer dependencies..."
$FORGE_COMPOSER install --no-dev --no-interaction --prefer-dist --optimize-autoloader -d $RELEASE_DIR

# Install Node.js dependencies and build frontend
echo "Installing Node.js dependencies..."
npm install
echo "Building frontend..."
npm run build
rm -rf node_modules

if [ -f artisan ]; then
    echo "Running migrations..."
    $FORGE_PHP artisan migrate --force
    
    echo "Running storage link..."
    $FORGE_PHP artisan storage:link --force
fi

# Switch symlink to new release
echo "Activating new release..."
ln -sfn $RELEASE_DIR/artisan $BASE_DIR/artisan 
ln -sfn $RELEASE_DIR $CURRENT_DIR

# Reload PHP-FPM
(
    flock -w 10 9 || exit 1
    echo "Restarting PHP-FPM..."
    sudo -S service $FORGE_PHP_FPM reload
) 9>/tmp/fpmlock

$FORGE_PHP artisan queue:restart

# Clean up old releases, keeping the latest 5
echo "Cleaning up old releases..."
cd $RELEASES_DIR
ls -1tr | head -n -5 | xargs -d '\n' rm -rf --



echo "Deployment completed successfully."