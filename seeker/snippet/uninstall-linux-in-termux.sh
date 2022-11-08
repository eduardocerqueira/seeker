#date: 2022-11-08T17:21:59Z
#url: https://api.github.com/gists/8831d1975c78edd8429bf6af6bbdd8bf
#owner: https://api.github.com/users/channthol

#!/bin/bash

echo ""
  echo "============================================="
  echo "========= Uninstall linux in Termux ========="
  echo "============================================="
  echo ""
echo ""

echo "==== Choose Uninstallation Option ===="
  distro=""
  echo "  1) Alpine"
  echo "  2) Debian"
  echo "  3) Fedora"
  echo "  4) Kali"
  echo "  5) Manjaro"
  echo "  6) Parrot"
  echo "  7) Ubuntu"
  echo "  8) Ubuntu19"
  echo "  9) Ubuntu20"
  echo "  10) Void"
  read o
  case $o in
    1) distro="Alpine";;
    2) distro="Debian";;
    3) distro="Fedora";;
    4) distro="Kali";;
    5) distro="Manjaro";;
    6) distro="Parrot";;
    7) distro="Ubuntu";;
    8) distro="Ubuntu19";;
    9) distro="Ubuntu20";;
    10) distro="Void";;
    *) echo "invalid option";exit;
  esac
  echo "《$distro selected》"

  installation_dir="$HOME/$distro"
  termux_bin="/data/data/com.termux/files/usr/bin/start-${distro,,}"

  if [ -d "$installation_dir" ]; then
        echo "==== Are you sure to delete 《$installation_dir》folder? ===="
        echo "  1) Yes"
        echo "  2) No"
        read d
        case $d in
                1)  echo "*** Deleting..please wait.."
                    rm -rf $installation_dir
                    if [ ! -d "$installation_dir" ]; then
                        echo "*** Folder Deleted successfully"
                        if [ -f "$termux_bin" ]; then
                           rm $termux_bin
                           if [ ! -f "$termux_bin" ]; then
                              echo "*** Start command Deleted successfully"
                              echo ""
                              echo "============================================="
                              echo "========= Uninstalled Successfully =========="
                              echo "============================================="
                              echo ""
                           fi
                        fi
                    fi
                ;;
                *)  echo ""
                    echo "==== Canceled uninstallation ===="
                    echo ""
                ;;
        esac
  else
        echo "Installation directory not found"
  fi