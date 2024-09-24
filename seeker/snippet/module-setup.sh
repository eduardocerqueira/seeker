#date: 2024-09-24T17:08:14Z
#url: https://api.github.com/gists/7f4209337062fb4db577887c3ff84c6c
#owner: https://api.github.com/users/arrjay

#!/usr/bin/env bash

# internal to get if the zfs root has a luks property
get_zfs_luksholder () {
    # we need to find the zfs dataset for root (if any)
    # this is somewhat duplicated from zfsexpandknowledge, but we want
    # abstract datasets, not devices.
    local mp
    local fstype
    local _
    local dataset
    local numfields="$(awk '{print NF; exit}' /proc/self/mountinfo)"
    if [[ "${numfields}" = "10" ]] ; then
        fields="_ _ _ _ mp _ _ fstype dev _"
    else
        fields="_ _ _ _ mp _ _ _ fstype dev _"
    fi
    # shellcheck disable=SC2086
    while read -r ${fields?} ; do
       [ "$fstype" = "zfs" ] || continue
       [ "$mp" = "/" ] && dataset="${dev}"
    done < /proc/self/mountinfo
    [[ "${dataset}" ]] || return 255

    # see if we have a luksholder property on that
    local luksholder="$(zfs get -H -o value io.github.arrjay:luksholder "${dataset}")"
    [[ "${luksholder}" ]] || return 255
    [[ "${luksholder}" == "-" ]] && return 255
    printf '%s' "${luksholder}"
}

check () {
    require_binaries mtype zfs cryptsetup "$systemdutildir"/systemd-cryptsetup

    [[ $hostonly ]] || [[ $mount_needs ]] && {
        case " ${host_fs_types[*]} " in
            *" zfs_member "*) : ;;
            *) return 255 ;;
        esac
        get_zfs_luksholder >/dev/null || return 255
    }
    return 0
}

# called by dracut
depends () {
    local deps
    echo "crypt zfs"
    return 0
}

# called by dracut
installkernel () {
    instmods dm_crypt
    # pry the kernel modules out of the luks header regardless of the mount status
    [[ $hostonly ]] && {
      local holder="$(get_zfs_luksholder)"
      local tdev="/dev/disk/by-uuid/${holder}"
      [[ -e "${tdev}" ]] || return 0
      local crypttype="$(cryptsetup luksDump "${tdev}"|awk '$1 ~ "cipher:" { split($2,o,"-") ; print o[1] ; }')"
      [[ "${crypttype}" ]] && instmods "crypto-${crypttype}"
    }
}

# called by dracut
install () {
    [[ "${hostonly}" ]] || return 0
    local holder="$(get_zfs_luksholder)"
    [[ "${holder}" ]] || return 0
    local escaped_holder="$(systemd-escape "luks-${holder}")"

    # wire cryptsetup...
    printf 'luks-%s UUID=%s none luks,nofail\n' "${holder}" "${holder}" >> "${initdir}/etc/crypttab"

    # create new target to run *after* zfs-import.target
    if dracut_module_included "systemd" ; then
        inst_simple "${moddir}/zfs-luksunwrap.target" "${systemdsystemunitdir}/zfs-luksunwrap.target"
        systemctl -q --root "${initdir}" add-wants initrd.target zfs-luksunwrap.target

        # add cryptsetup as a wants to our new target in a kinda clunky way
        mkdir -p "${initdir}/etc/systemd/system/systemd-cryptsetup@${escaped_holder}.service.d"
        {
          printf '[%s]\n' 'Unit'
          printf 'WantedBy=%s\n' 'zfs-luksunwrap.target'
        } > "${initdir}/etc/systemd/system/systemd-cryptsetup@${escaped_holder}.service.d/target.conf"
        mkdir -p "${initdir}/etc/systemd/system/zfs-luksunwrap.target.wants"
        # you can't do the below as it doesn't exist. but you can have drop-ins for units that are generated, as abpve...
        #systemctl -q --root "${initdir}" add-wants zfs-luksunwrap.target "systemd-cryptsetup@${escaped_holder}"
    fi

    # all the stuff needed for mktemp, sed, tr, grep, mtype, mdir - which our hook will call.
    local libtuple="$(dpkg-architecture -qDEB_BUILD_MULTIARCH)"
    inst_binary mktemp
    inst_binary sed
    inst_binary tr
    inst_binary grep
    inst_binary mtype
    inst_binary mdir
    inst_library "/usr/lib/${libtuple}/gconv/IBM850.so"
    inst_library "/usr/lib/${libtuple}/gconv/gconv-modules"
    [[ -e "/usr/lib/${libtuple}/gconv/gconv-modules.d/gconv-modules-extra.conf" ]] && inst_library "/usr/lib/${libtuple}/gconv/gconv-modules.d/gconv-modules-extra.conf"

    inst_hook pre-mount 89 "${moddir}/zfs-unwrap-luks.sh"
}