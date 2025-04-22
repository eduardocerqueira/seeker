#date: 2025-04-22T17:06:10Z
#url: https://api.github.com/gists/bc6dc1d95c5be9420d69e8e6188a0046
#owner: https://api.github.com/users/HSGEV

#!/bin/sh
#
# pve-bulk.sh - Copyright (c) 2018-2021 - Olivier Poncet
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

# ----------------------------------------------------------------------------
# command-line constants
# ----------------------------------------------------------------------------

arg_script=$(basename "${0:-not-set}")
arg_action='not-set'
arg_snapname='not-set'
opt_usage='no'
opt_error='no'

# ----------------------------------------------------------------------------
# pve constants
# ----------------------------------------------------------------------------

pve_ct_manager="${PVE_CT_MANAGER:-not-set}"
pve_ct_list="${PVE_CT_LIST:-not-set}"
pve_vm_manager="${PVE_VM_MANAGER:-not-set}"
pve_vm_list="${PVE_VM_LIST:-not-set}"
pve_min_id='100'
pve_max_id='9999'

# ----------------------------------------------------------------------------
# parse the action
# ----------------------------------------------------------------------------

if [ "${#}" -gt '0' ]
then
    arg_action="${1:-not-set}"
    shift
fi

# ----------------------------------------------------------------------------
# parse the positional parameters
# ----------------------------------------------------------------------------

case "${arg_action}" in
    'help')
        ;;
    'start')
        ;;
    'shutdown')
        ;;
    'stop')
        ;;
    'snapshot' | 'rollback' | 'delsnapshot')
        if [ "${#}" -gt '0' ]
        then
            arg_snapname="${1:-not-set}"
            shift
        fi
        if [ "${arg_snapname}" = 'not-set' ]
        then
            opt_error='yes'
        fi
        ;;
    *)
        opt_error='yes'
        ;;
esac

# ----------------------------------------------------------------------------
# parse the remaining options
# ----------------------------------------------------------------------------

while [ "${#}" -gt '0' ]
do
    case "${1}" in
        *=*)
            arg_value="$(expr "${1}" : '[^=]*=\(.*\)')"
            ;;
        *)
            arg_value=''
            ;;
    esac
    case "${1}" in
        --ct-list=*)
            pve_ct_list="${arg_value}"
            ;;
        --vm-list=*)
            pve_vm_list="${arg_value}"
            ;;
        --help)
            opt_usage='yes'
            ;;
        *)
            opt_error='yes'
            ;;
    esac
    shift
done

# ----------------------------------------------------------------------------
# display help if needed
# ----------------------------------------------------------------------------

if [ "${arg_action}" = 'help' ] || [ "${opt_usage}" = 'yes' ] || [ "${opt_error}" = 'yes' ]
then
    cat << ____EOF
Usage: ${arg_script} [ACTION [PARAMETERS]] [OPTIONS]

Actions:

    help
    start
    shutdown
    stop
    snapshot <snapname>
    rollback <snapname>
    delsnapshot <snapname>

Options:

    --help
    --ct-list={ctid,...}    ct list (defaults to pct list)
    --vm-list={vmid,...}    vm list (defaults to qm list)

Environment variables:

    PVE_CT_MANAGER          ct manager (defaults to pct)
    PVE_CT_LIST             ct list    (defaults to pct list)
    PVE_VM_MANAGER          vm manager (defaults to qm)
    PVE_VM_LIST             vm list    (defaults to qm list)

____EOF
    if [ "${opt_error}" = 'yes' ]
    then
        exit 1
    fi
    exit 0
fi

# ----------------------------------------------------------------------------
# pve_ct_manager
# ----------------------------------------------------------------------------

if [ "${pve_ct_manager}" = 'not-set' ]
then
    pve_ct_manager="$(which pct || echo 'not-found')"
fi

if [ "${pve_ct_manager}" = 'not-found' ]
then
    echo '*** pve_ct_manager was not found ***'
    exit 1
fi

# ----------------------------------------------------------------------------
# pve_vm_manager
# ----------------------------------------------------------------------------

if [ "${pve_vm_manager}" = 'not-set' ]
then
    pve_vm_manager="$(which qm || echo 'not-found')"
fi

if [ "${pve_vm_manager}" = 'not-found' ]
then
    echo '*** pve_vm_manager was not found ***'
    exit 1
fi

# ----------------------------------------------------------------------------
# pve_ct_list
# ----------------------------------------------------------------------------

if [ "${pve_ct_list}" = 'not-set' ]
then
    pve_ct_list=$(${pve_ct_manager} list | grep -v 'VMID' | awk '{ print $1 }')
else
    pve_ct_list=$(echo "${pve_ct_list}" | tr ',' ' ')
fi

# ----------------------------------------------------------------------------
# pve_vm_list
# ----------------------------------------------------------------------------

if [ "${pve_vm_list}" = 'not-set' ]
then
    pve_vm_list=$(${pve_vm_manager} list | grep -v 'VMID' | awk '{ print $1 }')
else
    pve_vm_list=$(echo "${pve_vm_list}" | tr ',' ' ')
fi

# ----------------------------------------------------------------------------
# ct:actions
# ----------------------------------------------------------------------------

for pve_ct_id in ${pve_ct_list}
do
    if [ "${pve_ct_id}" -lt "${pve_min_id}" ] || [ "${pve_ct_id}" -gt "${pve_max_id}" ]
    then
        continue
    fi
    case "ct:${arg_action}" in
        'ct:start')
            parameter1=""
            ${pve_ct_manager} "${arg_action}" "${pve_ct_id}" > /dev/null 2>&1
            status="${?}"
            ;;
        'ct:shutdown')
            parameter1=""
            ${pve_ct_manager} "${arg_action}" "${pve_ct_id}" > /dev/null 2>&1
            status="${?}"
            ;;
        'ct:stop')
            parameter1=""
            ${pve_ct_manager} "${arg_action}" "${pve_ct_id}" > /dev/null 2>&1
            status="${?}"
            ;;
        'ct:snapshot')
            parameter1="${arg_snapname}"
            ${pve_ct_manager} "${arg_action}" "${pve_ct_id}" "${parameter1}" > /dev/null 2>&1
            status="${?}"
            ;;
        'ct:rollback')
            parameter1="${arg_snapname}"
            ${pve_ct_manager} "${arg_action}" "${pve_ct_id}" "${parameter1}" > /dev/null 2>&1
            status="${?}"
            ;;
        'ct:delsnapshot')
            parameter1="${arg_snapname}"
            ${pve_ct_manager} "${arg_action}" "${pve_ct_id}" "${parameter1}" > /dev/null 2>&1
            status="${?}"
            ;;
        *)
            false
            ;;
    esac
    if [ "${status}" -eq '0' ]
    then
        result='succeeded'
    else
        result='failed'
    fi
    if [ "${parameter1:-not-set}" != 'not-set' ]
    then
        echo "ct${pve_ct_id} : ${arg_action} ${parameter1} has ${result}"
    else
        echo "ct${pve_ct_id} : ${arg_action} has ${result}"
    fi
done

# ----------------------------------------------------------------------------
# vm:actions
# ----------------------------------------------------------------------------

for pve_vm_id in ${pve_vm_list}
do
    if [ "${pve_vm_id}" -lt "${pve_min_id}" ] || [ "${pve_vm_id}" -gt "${pve_max_id}" ]
    then
        continue
    fi
    case "vm:${arg_action}" in
        'vm:start')
            parameter1=""
            ${pve_vm_manager} "${arg_action}" "${pve_vm_id}" > /dev/null 2>&1
            status="${?}"
            ;;
        'vm:shutdown')
            parameter1=""
            ${pve_vm_manager} "${arg_action}" "${pve_vm_id}" > /dev/null 2>&1
            status="${?}"
            ;;
        'vm:stop')
            parameter1=""
            ${pve_vm_manager} "${arg_action}" "${pve_vm_id}" > /dev/null 2>&1
            status="${?}"
            ;;
        'vm:snapshot')
            parameter1="${arg_snapname}"
            ${pve_vm_manager} "${arg_action}" "${pve_vm_id}" "${parameter1}" > /dev/null 2>&1
            status="${?}"
            ;;
        'vm:rollback')
            parameter1="${arg_snapname}"
            ${pve_vm_manager} "${arg_action}" "${pve_vm_id}" "${parameter1}" > /dev/null 2>&1
            status="${?}"
            ;;
        'vm:delsnapshot')
            parameter1="${arg_snapname}"
            ${pve_vm_manager} "${arg_action}" "${pve_vm_id}" "${parameter1}" > /dev/null 2>&1
            status="${?}"
            ;;
        *)
            false
            ;;
    esac
    if [ "${status}" -eq '0' ]
    then
        result='succeeded'
    else
        result='failed'
    fi
    if [ "${parameter1:-not-set}" != 'not-set' ]
    then
        echo "vm${pve_vm_id} : ${arg_action} ${parameter1} has ${result}"
    else
        echo "vm${pve_vm_id} : ${arg_action} has ${result}"
    fi
done

# ----------------------------------------------------------------------------
# End-Of-File
# ----------------------------------------------------------------------------
