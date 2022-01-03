#date: 2022-01-03T17:21:36Z
#url: https://api.github.com/gists/cdec4ef88fce5ccdce1f92bc64ff67b4
#owner: https://api.github.com/users/jevinskie

if type brew-x86 &>/dev/null; then
    export BREW_X86_ROOT=$(brew-x86 --prefix)
    # export PKG_CONFIG_PATH=${BREW_ROOT}/lib/pkgconfig
    # if [[ -r ${BREW_X86_ROOT}/etc/xml/catalog ]]; then
    #     export XML_CATALOG_FILES=${BREW_X86_ROOT}/etc/xml/catalog
    # fi
    function brew-pkgconfig-x86 {
        pkgconfig_path=""
        for pkg in $@; do
            pkgconfig_path+="${BREW_X86_ROOT}/opt/${pkg}/lib/pkgconfig:"
        done
        echo export PKG_CONFIG_PATH=${pkgconfig_path}${BREW_X86_ROOT}/lib/pkgconfig
    }
    function bprefix-x86 {
        echo "${BREW_X86_ROOT}/opt/$1"
    }
    function brew-is-installed-x86 {
        test -e "${BREW_X86_ROOT}/opt/$1"
        # if brew-x86 ls --versions $1 &> /dev/null; then
        #     return 0
        # fi
        # return 1
    }
    function brew-run-x86() {
        ${BREW_X86_ROOT}/opt/$1/bin/$2 ${@:3}
    }
fi