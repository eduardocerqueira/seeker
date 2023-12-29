#date: 2023-12-29T16:58:46Z
#url: https://api.github.com/gists/a1d127206051d2f9a21df6a71b5f85fd
#owner: https://api.github.com/users/korchasa

#!/usr/bin/env bash

# Check if at least two arguments are provided (Alpine version and one package)
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <alpine_version> <package1> [package2 ...]"
  exit 1
fi

alpine_version=$1
shift

docker run --rm alpine:$alpine_version sh -c "
    apk update
    output=''
    for package in \$@
    do
      # Extract the exact match of the package name and its latest version
      full_package=\$(apk search \${package} | grep -v -- '-doc' | grep -E \"^\${package}-[0-9]\" | sort -V | tail -n 1)
      if [ ! -z \"\${full_package}\" ]; then
        # Separate the package name from the version and suffix
        package_version=\$(echo \${full_package} | sed 's/^'\${package}'-//')
        output=\"\${output} \${package}=\${package_version}\"
      else
        echo \"Failed to find version for package \${package}\"
      fi
    done
    echo \$output
" -- "$@"
