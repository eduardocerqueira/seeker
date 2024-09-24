#date: 2024-09-24T17:08:14Z
#url: https://api.github.com/gists/7f4209337062fb4db577887c3ff84c6c
#owner: https://api.github.com/users/arrjay

#!/bin/sh

. /lib/dracut-zfs-lib.sh

# only activate if we're on zfs
# this should also set root to the dataset of note, I think?
decode_root_args || return 0

# see if we can get the prop off zfs now
get_zfsl_prop () {
    zfs_luksholder="$(zfs get -H -o value io.github.arrjay:luksholder "${root}")"
    [ "${zfs_luksholder}" ] || return 1
    [ "${zfs_luksholder}" = "-" ] && return 1
    zfs_luksholder_unit="systemd-cryptsetup@$(systemd-escape "luks-${zfs_luksholder}").service"
    zfs_luksholder="/dev/mapper/luks-${zfs_luksholder}"
}
get_zfsl_prop || return 0

# spin until we complete cryptsetup for our unit - which we should know here..
systemctl start "${zfs_luksholder_unit}" || return 0
while ! systemctl is-active --quiet "${zfs_luksholder_unit}" ; do
    # TODO: timeout?
    sleep 0.1s
done

zfsl_unlock () {
  # see if we have a blocky boy
  [ -e "${zfs_luksholder}" ] || return 0

  # get a list of the key files on the partition
  zfslklist="$(mktemp)"

  # deal with normalized key files (using dashes instead of /) and dataset components.
  zfsnorname="$(echo "${root}" | sed -e 's@/@-@g')"
  zfspooltarg="${root}"

  # grab all the keyfiles - we are expeting one level with dashes instead of slashes (if nested)
  mdir -f -b -i "${zfs_luksholder}" '*.key' | sed -e 's@::/@@g' -e 's@.key$@@g' > "${zfslklist}"

  # go looking, strip off datasets as we go up
  zfsfoundkey=0
  while [ "$(echo ${zfsnorname} | tr -cd '-')" != "" ] && [ "${zfsfoundkey}" -eq 0 ] ; do
    grep -q "${zfsnorname}" "${zfslklist}" && { zfsfoundkey=1 ; break ; }
    zfsnorname="${zfsnorname%-*}"
    zfspooltarg="${zfspooltarg%/*}"
  done
  # we need to do the test one last time in case we exhausted the loop
  [ "${zfsfoundkey}" -eq 0 ] && grep -q "${zfsnorname}" "${zfslklist}" && zfsfoundkey=1

  # try to load a key if we got one
  [ "${zfsfoundkey}" -eq 1 ] && mtype -i "${zfs_luksholder}" "::/${zfsnorname}.key" | zfs load-key "${zfspooltarg}"
}

zfsl_unlock

# unwind our crypt device while we're here but don't error
systemctl stop "${zfs_luksholder_unit}" || return 0