#date: 2026-02-25T17:53:44Z
#url: https://api.github.com/gists/4c672d77603f6d028d77aa0597f4cd51
#owner: https://api.github.com/users/danudey

#!/bin/bash

set -eu

DOWNLOAD_COMMAND=$(if which -s download; then echo download; else echo curl -fSLO --progress-bar; fi)

if [[ -v OC_VERSION ]]; then
    # Trim a preceeding `v` from version if it exists
    OC_VERSION="${OC_VERSION/#v/}"
    export KUBECTL_COMMAND="oc-orig-${OC_VERSION}"
    if ! which -s "${KUBECTL_COMMAND}"; then
        echo "Command for ${KUBECTL_COMMAND} not found, downloading..." >&2
        TEMPDIR=$(mktemp -d --tmpdir openshift-download.XXXXXXX)
        trap "rm -rf ${TEMPDIR}" EXIT
        # Switch to our temp dir
        cd "${TEMPDIR}"
        # Prune the patch version from the version string
        # SHORT_OC_VERSION="${OC_VERSION%.*}"
        tar_filename="openshift-client-linux-${OC_VERSION}.tar.gz"
        url="https://mirror.openshift.com/pub/openshift-v4/clients/ocp/${OC_VERSION}/${tar_filename}"
        ${DOWNLOAD_COMMAND} "${url}"
        tar xf "${tar_filename}" oc
        mv "${TEMPDIR}/oc" "${HOME}/.local/bin/oc-orig-${OC_VERSION}"
    fi
else
    export KUBECTL_COMMAND=oc-orig
fi


exec kubecolor "$@"
