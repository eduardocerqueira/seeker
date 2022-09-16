#date: 2022-09-16T22:01:41Z
#url: https://api.github.com/gists/fdc990550f0df033a008c3650bfe6f08
#owner: https://api.github.com/users/erlete

# Linux x64 GitHub Actions runner installation and configuration script.

ACTIONS_DIR_NAME="actions-runner"

# User acknowledgement prompt:

echo -n "[Warning] This script is only valid for Linux x64 architecture. Continue? (y/n) "
read -r answer
answer=$(echo "$answer" | tr '[:upper:]' '[:lower:]')

if [ -z "$answer" ]; then
    answer="y"
fi

if [ "$answer" != "y" ]; then
    echo "Aborting operation..."
    exit 1
fi

# Installation directory specification:

echo -n "Specify the installation directory (enter for default: \"$HOME\"): "
read -r install_dir

if [ -z "$install_dir" ]; then
    install_dir="$HOME"
else
    if [ ! -d "$install_dir" ]; then
        echo "[Error] The directory does not exist."
        exit 1
    fi
fi

OLDDIR=$(pwd)
CURDIR="$install_dir/$ACTIONS_DIR_NAME"
cd $CURDIR

# Installation/configuration process:

if [ ! -d "$CURDIR" ]; then

    # Environment creation:

    echo "[Log] Creating directory \"$ACTIONS_DIR_NAME\" on \"$install_dir\"..."
    mkdir -p $CURDIR

else
    echo "[Warning] The directory \"$ACTIONS_DIR_NAME\" already exists on \"$install_dir\". Attempting installation..."
fi

if [ ! -f "$CURDIR/actions-runner-linux-x64-2.296.2.tar.gz" ]; then

    # Latest runner package download:

    echo "[Log] Downloading latest runner package..."
    curl -o actions-runner-linux-x64-2.296.2.tar.gz -L https://github.com/actions/runner/releases/download/v2.296.2/actions-runner-linux-x64-2.296.2.tar.gz

else
    echo "[Warning] Runner package detected. Attempting SHA256 checksum verification..."
fi

# SHA256 checksum verification:

echo "[Log] Verifying SHA256 checksum..."
echo "34a8f34956cdacd2156d4c658cce8dd54c5aef316a16bbbc95eb3ca4fd76429a $CURDIR/actions-runner-linux-x64-2.296.2.tar.gz" | shasum -a 256 -c

# Installer extraction:

echo "[Log] Extracting installer..."
tar xzf $CURDIR/actions-runner-linux-x64-2.296.2.tar.gz

# Runner configuration:

$CURDIR/config.sh --url https: "**********"

# Execute the script:

$CURDIR/run.sh

# Return to the original directory:

cd "$OLDDIR" || exit 1
o the original directory:

cd "$OLDDIR" || exit 1
