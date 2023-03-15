#date: 2023-03-15T16:49:11Z
#url: https://api.github.com/gists/f2c541f6e027d7278e948f80e311e318
#owner: https://api.github.com/users/Gemba

#! /usr/bin/env bash

# This file is part of The RetroPie Project
#
# The RetroPie Project is the legal property of its developers, whose names are
# too numerous to list here. Please refer to the COPYRIGHT.md file distributed with this source.
#
# See the LICENSE.md file at the top-level directory of this distribution and
# at https://raw.githubusercontent.com/RetroPie/RetroPie-Setup/master/LICENSE.md
#

rp_module_id="hoh"
rp_module_desc="Open sourced and enhanced remake of 'Head over Heels'"
rp_module_licence="GPL3 https://github.com/dougmencken/HeadOverHeels/blob/master/LICENSE"
rp_module_repo="git https://github.com/dougmencken/HeadOverHeels.git"
rp_module_help="Batteries included: No extra gamefiles needed."
rp_module_section="exp"

function depends_hoh() {
    getDepends cmake liballegro4-dev libpng-dev libtinyxml2-dev libvorbis-dev rsync xorg
}

function sources_hoh() {
    gitPullOrClone
}

function build_hoh() {
    mkdir -p m4
    [ -f ./configure ] || autoreconf -f -i

    local gameInstallPath="$md_build"/_rootdir

    ./configure --with-allegro4 --prefix="${gameInstallPath}" --enable-debug=yes

    make clean
    make && make install
    md_ret_require="$md_build/_rootdir/bin/headoverheels"
}

function install_hoh() {
    local game_dest="$romdir/ports/headoverheels"
    for d in bin share ; do
        mkUserDir "$game_dest/$d"
        rsync --recursive --owner --group --chown="$user":"$user" \
            "$md_build"/_rootdir/$d/* "$game_dest/$d"
    done
    cp -p "$md_build"/LICENSE "$game_dest"
    chown "$user":"$user" "$game_dest"/LICENSE
}

function configure_hoh() {

    local ports_cfg_dir="headoverheels"

    addPort "$md_id" "$ports_cfg_dir" "Head over Heels" "XINIT:/opt/retropie/ports/hoh/launcher.sh"
    [[ "$md_mode" != "install" ]] && return

    local pref_dir="$home"/.headoverheels
    local pref_file="$pref_dir"/preferences.xml

    mkUserDir "$pref_dir"
    moveConfigDir "$pref_dir" "$md_conf_root/$ports_cfg_dir"

    if [[ ! -e "$pref_file" ]] ; then
        cat >"$pref_file" << _EOF_
<preferences>
    <language>en_US</language>
    <keyboard>
        <movenorth>Left</movenorth>
        <movesouth>Right</movesouth>
        <moveeast>Up</moveeast>
        <movewest>Down</movewest>
        <jump>Space</jump>
        <take>c</take>
        <takeandjump>b</takeandjump>
        <doughnut>n</doughnut>
        <swap>x</swap>
        <pause>m</pause>
        <automap>Tab</automap>
    </keyboard>
    <audio>
        <fx>80</fx>
        <music>75</music>
        <roomtunes>true</roomtunes>
    </audio>
    <video>
        <fullscreen>true</fullscreen>
        <shadows>true</shadows>
        <background>true</background>
        <graphics>gfx</graphics>
    </video>
</preferences>
_EOF_
        chown "$user": "$pref_file"
    fi

    cat >"$md_inst"/launcher.sh << _EOF_
#! /usr/bin/env bash
xset -dpms s off s noblank
cd "$romdir/ports/headoverheels"
bin/headoverheels
killall xinit
_EOF_

    chown "$user": "$md_inst"/launcher.sh
    chmod a+x "$md_inst"/launcher.sh
}
