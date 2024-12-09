#date: 2024-12-09T17:05:29Z
#url: https://api.github.com/gists/dda8ab906f3b1ebd787938ad3be6809d
#owner: https://api.github.com/users/iancaseydouglas

#!/bin/bash

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--source)
                SOURCE_REPO="$2"
                shift 2
                ;;
            -d|--destination)
                DEST_REPO="$2"
                shift 2
                ;;
            -q|--quiet)
                VERBOSE=false
                shift
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
}

ensure_trailing_slash() {
    local path=$1
    [[ "${path}" != */ ]] && path="${path}/"
    echo "$path"
}

sync_repos() {
    local source_path=$(ensure_trailing_slash "$1")
    local dest_path=$(ensure_trailing_slash "$2")
    local verbose=$3

    local rsync_opts="-ah"
    if [ "$verbose" = true ]; then
        rsync_opts="${rsync_opts}v"
    fi

    rsync ${rsync_opts}u \
        --exclude=".git/" \
        "${source_path}" \
        "${dest_path}"
}

main() {
    SOURCE_REPO=""
    DEST_REPO=""
    VERBOSE=true

    parse_arguments "$@"

    if [ -z "$SOURCE_REPO" ] || [ -z "$DEST_REPO" ]; then
        echo "Usage: $0 -s|--source <source_path> -d|--destination <dest_path> [-q|--quiet]"
        exit 1
    fi

    sync_repos "$SOURCE_REPO" "$DEST_REPO" "$VERBOSE"
    echo "Sync complete!"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
