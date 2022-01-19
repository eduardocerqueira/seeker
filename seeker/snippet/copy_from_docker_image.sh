#date: 2022-01-19T16:52:07Z
#url: https://api.github.com/gists/79e9f500e699361eea543abcd1df3111
#owner: https://api.github.com/users/barn

#!/bin/zsh
#
# Supports <image>:<path>, supports sha256/tags, path supports wildards, if a second arg is a path, it will output there, creating paths if needed
#   $ copy_from_docker_image ruby:2.7.4:/etc/hosts ./

copy_from_docker_image() {
    if [ -z "$ZSH_VERSION" ]; then
        echo "sorry I need to use zsh"
        exit 1
    fi
    
    setopt local_options pipefail errexit nounset
    local image_arg _filename _image output_path
    typeset -a bits _f lessbits
    image_arg="$1"
    output_path="${2-.}"

    # If the target directory doesn't exist, try to make it, fail
    # otherwise
    [ -d "$output_path" ] || mkdir -p "$output_path" || { echo "cannot make output dir" ; exit 20; }

    # split string by :
    bits=("${(@s/:/)image_arg}")

    # Get the last element of the split array
    _filename="${bits[-1]}"
    _f=($_filename)  # make a temp array just with it.

    # remove _f from the array
    lessbits=(${bits:|_f})
    # join them back together with ':'
    _image="${(j(:))lessbits}"

    # dirname and basename in zsh
    cdpath=${_filename:h}
    files=${_filename:t}

    # Run the container, but with as many options to increase safety as
    # possible. No networking, no caps, read-only, etc etc
    # We're trusting the tar on the target host sadly.
    docker run \
        --pull missing \
        --user 0:0 \
        --no-healthcheck \
        --read-only \
        --cap-drop=all \
        --security-opt=no-new-privileges:true \
        --network none \
        --entrypoint /bin/sh \
        --rm \
        "$_image" -c "tar -cf - -C \"${cdpath}\" ${files}" | tar xvvf - -C "$output_path"

    # No one has ever gotten pv to be helpful.
    # | ( if [ -n "$commands[pv]" ]; then pv -tr -B 1024 -W --cursor | tar xvvf - -C "$output_path" >&4 ; else tar xvvf - -C "$output_path" ; fi )

}