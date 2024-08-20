#date: 2024-08-20T16:45:31Z
#url: https://api.github.com/gists/ff44e7ad3f67fc5b59a5fde56ba0f1ef
#owner: https://api.github.com/users/ankitahuja0508

# Function to check if a value is an integer
is_integer() {
    [ "$1" -eq "$1" ] 2>/dev/null
}

# Remove any existing JAVA_HOME entries in .zshrc
echo "Removing old JAVA_HOME if any..."
sed -i '' '/export JAVA_HOME=/d' ~/.zshrc

# Check if an argument is passed (the Java version)
if [ -n "$1" ]; then
    # Check if the argument is an integer
    if is_integer "$1"; then
        echo "Installing Oracle JDK version $1..."
        brew install oracle-jdk@$1 --cask
        echo "setting JAVA_HOME for version $1..."
        echo export "JAVA_HOME=\$(/usr/libexec/java_home -v $1)" >> ~/.zshrc
    else
        echo "Error: The argument '$1' is not an integer. Please provide a valid Java version number." >&2
        exit 1
    fi
else
    echo "Installing the latest Oracle JDK..."
    brew install oracle-jdk --cask
    echo "setting JAVA_HOME for the latest version..."
    echo export "JAVA_HOME=\$(/usr/libexec/java_home)" >> ~/.zshrc
fi

# Apply changes to .zshrc
source ~/.zshrc

# Check Java version
echo "checking java version"
java -version