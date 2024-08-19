#date: 2024-08-19T16:53:21Z
#url: https://api.github.com/gists/3560f17bc52bede36670527a00b14f88
#owner: https://api.github.com/users/OwOchle

#! /bin/bash

cleanup() {
    echo "> cleaning up";
    rm -rf $1;
    git checkout $BEFORE_CHECKOUT;
}

warn() {
    echo -e "\e[1;33m$1\e[0m"
}

exit_no_cleanup() {
    echo -e "\e[31m$1\e[0m" > /dev/stderr;
    exit 1;
}

exit_cleanup() {
    echo -e "\e[31m$1\e[0m" > /dev/stderr;
    cleanup $2;
    exit 1;
}

ask_confirm() {
    read -p "$1? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        echo "aborting"
        exit 1
    fi
}

echo "===== WARNING ====="
echo "In order to ensure correct version of submodules, this utility will checkout the provided revision"
ask_confirm "Continue"

BEFORE_CHECKOUT=$(git rev-parse --short HEAD)

if ! git status > /dev/null 2> /dev/null;
then
    exit_no_cleanup "> not a git repository";
fi


if [ -n "$2" ];
then
    echo "> checking existence of revision $2"

    if ! git log "$2" > /dev/null 2> /dev/null;
    then
        exit_no_cleanup "> revision $2 not found"
    fi

    COMMIT=$2
else
    warn "> no revision provided, defaulting to HEAD"
    COMMIT="HEAD"
fi

echo "> checking out revision"
git checkout "$COMMIT"

echo "> updating submodules"
git submodule update --init --recursive

echo "> creating root archive"
TEMPORARY=$(mktemp -d)


if ! git archive --prefix "/" --format "tar" --output "$TEMPORARY/repo-output.tar" "$COMMIT" ;
then
    echo "> error archiving base repo" > /dev/stderr
    cleanup "$TEMPORARY"
    exit 1
fi

echo "> creating submodule archives"

GIT_SUBCOMMAND='git archive --prefix=/$path/ --format tar HEAD --output'

if ! git submodule foreach --recursive "$GIT_SUBCOMMAND $TEMPORARY/repo-output-sub-\$sha1.tar";
then
    exit_cleanup "> error while archiving submodules" $TEMPORARY
fi

ls "$TEMPORARY"

if [[ $(find "$TEMPORARY" -iname "repo-output-sub*.tar" | wc -l) != 0  ]];
then
    echo "> combining tars"
    tar --concatenate --file "$TEMPORARY/repo-output.tar" "$TEMPORARY"/repo-output-sub*.tar
fi

if [ "$1" = "" ] || [ "$1" = "-" ];
then
    OUTPUT="$(basename "$(git rev-parse --show-toplevel)").tar.zstd"
else
    OUTPUT="$1"
fi

echo "> zstd compression to $OUTPUT"
zstd -q -f -T0 "$TEMPORARY/repo-output.tar" -o "$OUTPUT"
cleanup "$TEMPORARY"