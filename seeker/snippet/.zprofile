#date: 2023-08-16T17:07:46Z
#url: https://api.github.com/gists/ef9aeef71579d97796e7c58aa48748b1
#owner: https://api.github.com/users/muh-dirga-f

openjdk() {
    local version=$1

    case $version in
        8 | 11)
            local gradle_version=6
            ;;
        17)
            local gradle_version=7
            ;;
        latest)
            local gradle_version=""
            ;;
        *)
            echo "Versi OpenJDK tidak didukung: $version"
            return 1
            ;;
    esac

    if [ -n "$gradle_version" ]; then
        export PATH="/usr/local/opt/openjdk@$version/bin:$PATH"
        export JAVA_HOME="/usr/local/opt/openjdk@$version"
        export PATH="/usr/local/opt/gradle@$gradle_version/bin:$PATH"
    else
        export PATH="/usr/local/opt/openjdk/bin:$PATH"
        export JAVA_HOME="/usr/local/opt/openjdk"
        export PATH="/usr/local/opt/gradle/bin:$PATH"
    fi

    # print version
    echo "------------------------------------------------------------"
    echo "OpenJDK Status"
    echo "------------------------------------------------------------"
    java -version
    gradle -v
}