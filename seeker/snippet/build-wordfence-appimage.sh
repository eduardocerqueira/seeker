#date: 2024-04-23T16:50:42Z
#url: https://api.github.com/gists/e155af1c529d3cc43b76f6055bcf2ddc
#owner: https://api.github.com/users/zsteva

#!/bin/bash

# wordfence appimage builder
# by zsteva@gmail.com

APPTOOL_URL="https://github.com/AppImage/AppImageKit/releases/download/13/appimagetool-x86_64.AppImage"
PY_URL="https://github.com/niess/python-appimage/releases/download/python3.12/python3.12.3-cp312-cp312-manylinux_2_28_x86_64.AppImage"

APPTOOL_NAME="$(basename "$APPTOOL_URL")"
PY_NAME="$(basename "$PY_URL")"

if [ ! -e "${APPTOOL_NAME}" ]; then
    rm -f "${APPTOOL_NAME}_"

    wget -O "${APPTOOL_NAME}_" "${APPTOOL_URL}" \
        && mv "${APPTOOL_NAME}_" "${APPTOOL_NAME}" \
        || exit 39

    chmod a+rx "${APPTOOL_NAME}"
fi

if [ ! -e "${PY_NAME}" ]; then
    rm -f "${PY_NAME}_"

    wget -O "${PY_NAME}_" "${PY_URL}" \
        && mv "${PY_NAME}_" "${PY_NAME}" \
        || exit 39

    chmod a+rx "${PY_NAME}"
fi

rm -rf "./squashfs-root"

./${PY_NAME} --appimage-extract

./squashfs-root/AppRun -m pip install wordfence

rm -f ./squashfs-root/*.desktop

cat > ./squashfs-root/wordfence.desktop  << __EOF__
[Desktop Entry]
Type=Application
Name=wordfence
Exec=wordfence
Comment=wordfence
Icon=python
Categories=Utility;
Terminal=true
__EOF__

sed -i -e 's,"$@","${APPDIR}/usr/bin/wordfence" "$@",' squashfs-root/AppRun

rm -f "wordfence.AppImage"

./${APPTOOL_NAME} ./squashfs-root/ wordfence.AppImage

rm -rf "./squashfs-root"

