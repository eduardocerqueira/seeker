#date: 2024-03-08T17:00:56Z
#url: https://api.github.com/gists/046449a4a3efc932e161d2686a3ef3f5
#owner: https://api.github.com/users/marcransome

#!/bin/bash

set +o pipefail

if (( $# != 1 )); then
    echo "usage: $0 <ref>" >&2
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    echo "error: git executable not found" >&2
    exit 1
fi

ref="$1"
obj=$(git rev-parse "$ref" 2>/dev/null)
if (( $? != 0 )); then
    echo "error: unknown revision or not a git repository" >&2
    exit 1
fi

while true; do
    readarray -d ' ' -t tokens <<< "$(git cat-file -p "$obj" 2>/dev/null | head -1)"
    if (( $? != "**********"
        echo "error: unable to determine details of object: $obj" >&2
        exit 1
    fi
    
    type=$(git cat-file -t "$obj" 2>/dev/null)
    if (( $? != 0 )); then
        echo "error: unable to obtain content of object: $obj" >&2 
        exit 1
    fi

    node="${node:+ -> }[ $type:${obj:0:6} ]"
    echo -n "$node"
    
    case "${tokens[0]}" in
        object|tree)
            obj= "**********"
            ;;
        *)
            echo
            break
            ;;
    esac

done
    break
            ;;
    esac

done
