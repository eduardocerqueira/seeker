#date: 2024-02-16T16:56:51Z
#url: https://api.github.com/gists/478874632fca61869928a0cc0a956972
#owner: https://api.github.com/users/ole

#!/bin/zsh

# Test if the Swift compiler knows about a particular language feature.
#
# Usage:
#
#     swift-has-feature [--swift SWIFT_PATH] FEATURE
#
# The exit code signals success (= the compiler knows this feature) or failure.
#
# Example:
#
#     swift-has-feature FullTypedThrows
#     swift-has-feature --swift /Library/Developer/Toolchains/swift-DEVELOPMENT-SNAPSHOT-2024-01-04-a.xctoolchain/usr/bin/swift FullTypedThrows
#
# The feature be an upcoming or experimental language feature,
# such as `"StrictConcurrency"` or `"FullTypedThrows"`.
#
# Unfortunately, the script can’t tell you whether the feature in question
# is upcoming or experimental or neither. Use swift-list-features.sh for that.

usage="swift-has-feature [--swift <path_to_swift>] [--silent] <feature>"

zmodload zsh/zutil
zparseopts -D -F -- -swift:=arg_swift_path -silent=flag_silent || exit 1

if test -z "$1"; then
    echo "Usage: $usage"
    exit
fi

swift_path=${arg_swift_path[-1]}
if test -z "$swift_path"; then
    swift_path="swift"
fi

# Print compiler version
if test -z "$flag_silent"; then
    "$swift_path" --version
fi

"$swift_path" \
    -enable-upcoming-feature "$1" \
    -enable-experimental-feature "$1" \
    - << END_OF_SWIFT_SCRIPT
import Foundation

#if hasFeature($1)
    exit(EXIT_SUCCESS)
#else
    exit(EXIT_FAILURE)
#endif
END_OF_SWIFT_SCRIPT

if test $? -eq 0; then
    if test -z "$flag_silent"; then
        echo "Supported: Swift compiler knows feature '$1'"
    fi
else
    if test -z "$flag_silent"; then
        echo "Not supported: Swift compiler doesn’t know feature '$1'"
    fi
    exit 1
fi