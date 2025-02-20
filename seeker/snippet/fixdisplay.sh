#date: 2025-02-20T16:45:06Z
#url: https://api.github.com/gists/a1c3b86fab4e51f4b9e406a8edf05525
#owner: https://api.github.com/users/bartmeuris

#!/bin/bash
# Script to align an externally connected display on your mac to be automatically positioned and centered above the laptop display.
# Requires the displayplacer tool (https://github.com/jakehilborn/displayplacer) with the JSON PR (https://github.com/jakehilborn/displayplacer/pull/143) merged.

TMPF=$(mktemp)
displayplacer list --json > ${TMPF}
# Remove temp file after exit with trap
cleanup() {
    rm -f "${TMPF}"
}

trap cleanup SIGINT SIGTERM

EXT_DISPLAY_CNT=$(cat ${TMPF} | jq -r '.screens[] | select( .type != "MacBook built in screen" ) | .persistent_screen_id' | wc -l)

if [ "${EXT_DISPLAY_CNT}" -ne 1 ]; then
    echo "Expected one external display, got ${EXT_DISPLAY_CNT}"
    exit 1
fi
INTERNAL_DISPLAY=$(cat ${TMPF} | jq -r '.screens[] | select( .builtin ) | .persistent_screen_id')
EXTERNAL_DISPLAY=$(cat ${TMPF} | jq -r '.screens[] | select( .type != "MacBook built in screen" ) | .persistent_screen_id')

origin() {
    DISPLAY=$1
    ORIGIN_X=$(cat ${TMPF} | jq -r '.screens[] | select( .persistent_screen_id == "'${DISPLAY}'" ) | .origin.x')
    ORIGIN_Y=$(cat ${TMPF} | jq -r '.screens[] | select( .persistent_screen_id == "'${DISPLAY}'" ) | .origin.y')
    echo "origin:(${ORIGIN_X},${ORIGIN_Y})"
}

centerorigin() {
    DISPLAY_CENTER=$1
    DISPLAY_MAIN=$2
    
    RX=$(cat ${TMPF} | jq -r '.screens[] | select( .persistent_screen_id == "'${DISPLAY_MAIN}'" ) | .resolution.x')
    DX=$(cat ${TMPF} | jq -r '.screens[] | select( .persistent_screen_id == "'${DISPLAY_CENTER}'" ) | .resolution.x')
    CX=$(( (( $DX - $RX ) / 2 ) * -1 ))
    CY="-$(cat ${TMPF} | jq -r '.screens[] | select( .persistent_screen_id == "'${DISPLAY_CENTER}'" ) | .resolution.y')"

    echo "origin:(${CX},${CY})"
}

dispparam() {
    DISPLAY=$1
    ORIGIN=$2
    [ -z "${ORIGIN}" ] && ORIGIN=$(origin $DISPLAY)
    RES_X=$(cat ${TMPF} | jq -r '.screens[] | select( .persistent_screen_id == "'${DISPLAY}'" ) | .resolution.x')
    RES_Y=$(cat ${TMPF} | jq -r '.screens[] | select( .persistent_screen_id == "'${DISPLAY}'" ) | .resolution.y')
    RES="${RES_X}x${RES_Y}"
    HZ=$(cat ${TMPF} | jq -r '.screens[] | select( .persistent_screen_id == "'${DISPLAY}'" ) | .hertz')
    COLOR=$(cat ${TMPF} | jq -r '.screens[] | select( .persistent_screen_id == "'${DISPLAY}'" ) | .color_depth')
    SCALING=$(cat ${TMPF} | jq -r '.screens[] | select( .persistent_screen_id == "'${DISPLAY}'" ) | .scaling')

    ENABLED=$(cat ${TMPF} | jq -r '.screens[] | select( .persistent_screen_id == "'${DISPLAY}'" ) | .enabled')
    
    echo "id:${DISPLAY} res:${RES} hz:${HZ} color_depth:${COLOR} enabled:${ENABLED} scaling:${SCALING} ${ORIGIN} degree:0"
}

DISP1=$(dispparam ${INTERNAL_DISPLAY})
DISP2=$(dispparam ${EXTERNAL_DISPLAY} $(centerorigin ${EXTERNAL_DISPLAY} ${INTERNAL_DISPLAY}))
displayplacer "${DISP1}" "${DISP2}"
cleanup
