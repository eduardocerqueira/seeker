#date: 2022-11-15T17:08:49Z
#url: https://api.github.com/gists/0dd6f77b89df5147e238fd7d52415907
#owner: https://api.github.com/users/fabiolimace

#!/bin/bash
#
# Decodes percent-encoded or URL-encoded strings to UTF-8
#
# How to use:
#
#    ./percent-decode.sh STRING
#
# Example:
#
#    ./percent-decode.sh "https://pt.wikipedia.org/wiki/Codifica%C3%A7%C3%A3o_por_cento"
#                 OUTPUT: https://pt.wikipedia.org/wiki/Codificação_por_cento
#
# Author: Fabio Lima
# Created: 2022-11-15
#
# See: [RFC-3986 - 2.1 Percent-Encoding](https://www.rfc-editor.org/rfc/rfc3986#section-2.1)
#

function decode {

    INPUT="${1}"
    OUTPUT=""
    
    for OCTECT in `echo "$INPUT" | grep -E --only-matching "([^%]|%[0-9A-F]{2})"`; do
    
        if [[ "${OCTECT}" =~ %[0-9A-F]{2} ]]; then
            OUTPUT="${OUTPUT}`echo $OCTECT | tr -d '%' | xxd -p -r`"
        else
            OUTPUT="${OUTPUT}${OCTECT}"
        fi;
    done;
    
    echo -n "${OUTPUT}"
}

decode "${1}"

