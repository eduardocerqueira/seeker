#date: 2025-05-01T16:58:52Z
#url: https://api.github.com/gists/7aad9064c2fba0472e1a65c8cc983b11
#owner: https://api.github.com/users/sahapasci

# bash completion for etcdctl                              -*- shell-script -*-

__etcdctl_debug()
{
    if [[ -n ${BASH_COMP_DEBUG_FILE:-} ]]; then
        echo "$*" >> "${BASH_COMP_DEBUG_FILE}"
    fi
}

# Homebrew on Macs have version 1.3 of bash-completion which doesn't include
# _init_completion. This is a very minimal version of that function.
__etcdctl_init_completion()
{
    COMPREPLY=()
    _get_comp_words_by_ref "$@" cur prev words cword
}

__etcdctl_index_of_word()
{
    local w word=$1
    shift
    index=0
    for w in "$@"; do
        [[ $w = "$word" ]] && return
        index=$((index+1))
    done
    index=-1
}

__etcdctl_contains_word()
{
    local w word=$1; shift
    for w in "$@"; do
        [[ $w = "$word" ]] && return
    done
    return 1
}

__etcdctl_handle_go_custom_completion()
{
    __etcdctl_debug "${FUNCNAME[0]}: cur is ${cur}, words[*] is ${words[*]}, #words[@] is ${#words[@]}"

    local shellCompDirectiveError=1
    local shellCompDirectiveNoSpace=2
    local shellCompDirectiveNoFileComp=4
    local shellCompDirectiveFilterFileExt=8
    local shellCompDirectiveFilterDirs=16

    local out requestComp lastParam lastChar comp directive args

    # Prepare the command to request completions for the program.
    # Calling ${words[0]} instead of directly etcdctl allows handling aliases
    args=("${words[@]:1}")
    # Disable ActiveHelp which is not supported for bash completion v1
    requestComp="ETCDCTL_ACTIVE_HELP=0 ${words[0]} __completeNoDesc ${args[*]}"

    lastParam=${words[$((${#words[@]}-1))]}
    lastChar=${lastParam:$((${#lastParam}-1)):1}
    __etcdctl_debug "${FUNCNAME[0]}: lastParam ${lastParam}, lastChar ${lastChar}"

    if [ -z "${cur}" ] && [ "${lastChar}" != "=" ]; then
        # If the last parameter is complete (there is a space following it)
        # We add an extra empty parameter so we can indicate this to the go method.
        __etcdctl_debug "${FUNCNAME[0]}: Adding extra empty parameter"
        requestComp="${requestComp} \"\""
    fi

    __etcdctl_debug "${FUNCNAME[0]}: calling ${requestComp}"
    # Use eval to handle any environment variables and such
    out=$(eval "${requestComp}" 2>/dev/null)

    # Extract the directive integer at the very end of the output following a colon (:)
    directive=${out##*:}
    # Remove the directive
    out=${out%:*}
    if [ "${directive}" = "${out}" ]; then
        # There is not directive specified
        directive=0
    fi
    __etcdctl_debug "${FUNCNAME[0]}: the completion directive is: ${directive}"
    __etcdctl_debug "${FUNCNAME[0]}: the completions are: ${out}"

    if [ $((directive & shellCompDirectiveError)) -ne 0 ]; then
        # Error code.  No completion.
        __etcdctl_debug "${FUNCNAME[0]}: received error from custom completion go code"
        return
    else
        if [ $((directive & shellCompDirectiveNoSpace)) -ne 0 ]; then
            if [[ $(type -t compopt) = "builtin" ]]; then
                __etcdctl_debug "${FUNCNAME[0]}: activating no space"
                compopt -o nospace
            fi
        fi
        if [ $((directive & shellCompDirectiveNoFileComp)) -ne 0 ]; then
            if [[ $(type -t compopt) = "builtin" ]]; then
                __etcdctl_debug "${FUNCNAME[0]}: activating no file completion"
                compopt +o default
            fi
        fi
    fi

    if [ $((directive & shellCompDirectiveFilterFileExt)) -ne 0 ]; then
        # File extension filtering
        local fullFilter filter filteringCmd
        # Do not use quotes around the $out variable or else newline
        # characters will be kept.
        for filter in ${out}; do
            fullFilter+="$filter|"
        done

        filteringCmd="_filedir $fullFilter"
        __etcdctl_debug "File filtering command: $filteringCmd"
        $filteringCmd
    elif [ $((directive & shellCompDirectiveFilterDirs)) -ne 0 ]; then
        # File completion for directories only
        local subdir
        # Use printf to strip any trailing newline
        subdir=$(printf "%s" "${out}")
        if [ -n "$subdir" ]; then
            __etcdctl_debug "Listing directories in $subdir"
            __etcdctl_handle_subdirs_in_dir_flag "$subdir"
        else
            __etcdctl_debug "Listing directories in ."
            _filedir -d
        fi
    else
        while IFS='' read -r comp; do
            COMPREPLY+=("$comp")
        done < <(compgen -W "${out}" -- "$cur")
    fi
}

__etcdctl_handle_reply()
{
    __etcdctl_debug "${FUNCNAME[0]}"
    local comp
    case $cur in
        -*)
            if [[ $(type -t compopt) = "builtin" ]]; then
                compopt -o nospace
            fi
            local allflags
            if [ ${#must_have_one_flag[@]} -ne 0 ]; then
                allflags=("${must_have_one_flag[@]}")
            else
                allflags=("${flags[*]} ${two_word_flags[*]}")
            fi
            while IFS='' read -r comp; do
                COMPREPLY+=("$comp")
            done < <(compgen -W "${allflags[*]}" -- "$cur")
            if [[ $(type -t compopt) = "builtin" ]]; then
                [[ "${COMPREPLY[0]}" == *= ]] || compopt +o nospace
            fi

            # complete after --flag=abc
            if [[ $cur == *=* ]]; then
                if [[ $(type -t compopt) = "builtin" ]]; then
                    compopt +o nospace
                fi

                local index flag
                flag="${cur%=*}"
                __etcdctl_index_of_word "${flag}" "${flags_with_completion[@]}"
                COMPREPLY=()
                if [[ ${index} -ge 0 ]]; then
                    PREFIX=""
                    cur="${cur#*=}"
                    ${flags_completion[${index}]}
                    if [ -n "${ZSH_VERSION:-}" ]; then
                        # zsh completion needs --flag= prefix
                        eval "COMPREPLY=( \"\${COMPREPLY[@]/#/${flag}=}\" )"
                    fi
                fi
            fi

            if [[ -z "${flag_parsing_disabled}" ]]; then
                # If flag parsing is enabled, we have completed the flags and can return.
                # If flag parsing is disabled, we may not know all (or any) of the flags, so we fallthrough
                # to possibly call handle_go_custom_completion.
                return 0;
            fi
            ;;
    esac

    # check if we are handling a flag with special work handling
    local index
    __etcdctl_index_of_word "${prev}" "${flags_with_completion[@]}"
    if [[ ${index} -ge 0 ]]; then
        ${flags_completion[${index}]}
        return
    fi

    # we are parsing a flag and don't have a special handler, no completion
    if [[ ${cur} != "${words[cword]}" ]]; then
        return
    fi

    local completions
    completions=("${commands[@]}")
    if [[ ${#must_have_one_noun[@]} -ne 0 ]]; then
        completions+=("${must_have_one_noun[@]}")
    elif [[ -n "${has_completion_function}" ]]; then
        # if a go completion function is provided, defer to that function
        __etcdctl_handle_go_custom_completion
    fi
    if [[ ${#must_have_one_flag[@]} -ne 0 ]]; then
        completions+=("${must_have_one_flag[@]}")
    fi
    while IFS='' read -r comp; do
        COMPREPLY+=("$comp")
    done < <(compgen -W "${completions[*]}" -- "$cur")

    if [[ ${#COMPREPLY[@]} -eq 0 && ${#noun_aliases[@]} -gt 0 && ${#must_have_one_noun[@]} -ne 0 ]]; then
        while IFS='' read -r comp; do
            COMPREPLY+=("$comp")
        done < <(compgen -W "${noun_aliases[*]}" -- "$cur")
    fi

    if [[ ${#COMPREPLY[@]} -eq 0 ]]; then
        if declare -F __etcdctl_custom_func >/dev/null; then
            # try command name qualified custom func
            __etcdctl_custom_func
        else
            # otherwise fall back to unqualified for compatibility
            declare -F __custom_func >/dev/null && __custom_func
        fi
    fi

    # available in bash-completion >= 2, not always present on macOS
    if declare -F __ltrim_colon_completions >/dev/null; then
        __ltrim_colon_completions "$cur"
    fi

    # If there is only 1 completion and it is a flag with an = it will be completed
    # but we don't want a space after the =
    if [[ "${#COMPREPLY[@]}" -eq "1" ]] && [[ $(type -t compopt) = "builtin" ]] && [[ "${COMPREPLY[0]}" == --*= ]]; then
       compopt -o nospace
    fi
}

# The arguments should be in the form "ext1|ext2|extn"
__etcdctl_handle_filename_extension_flag()
{
    local ext="$1"
    _filedir "@(${ext})"
}

__etcdctl_handle_subdirs_in_dir_flag()
{
    local dir="$1"
    pushd "${dir}" >/dev/null 2>&1 && _filedir -d && popd >/dev/null 2>&1 || return
}

__etcdctl_handle_flag()
{
    __etcdctl_debug "${FUNCNAME[0]}: c is $c words[c] is ${words[c]}"

    # if a command required a flag, and we found it, unset must_have_one_flag()
    local flagname=${words[c]}
    local flagvalue=""
    # if the word contained an =
    if [[ ${words[c]} == *"="* ]]; then
        flagvalue=${flagname#*=} # take in as flagvalue after the =
        flagname=${flagname%=*} # strip everything after the =
        flagname="${flagname}=" # but put the = back
    fi
    __etcdctl_debug "${FUNCNAME[0]}: looking for ${flagname}"
    if __etcdctl_contains_word "${flagname}" "${must_have_one_flag[@]}"; then
        must_have_one_flag=()
    fi

    # if you set a flag which only applies to this command, don't show subcommands
    if __etcdctl_contains_word "${flagname}" "${local_nonpersistent_flags[@]}"; then
      commands=()
    fi

    # keep flag value with flagname as flaghash
    # flaghash variable is an associative array which is only supported in bash > 3.
    if [[ -z "${BASH_VERSION:-}" || "${BASH_VERSINFO[0]:-}" -gt 3 ]]; then
        if [ -n "${flagvalue}" ] ; then
            flaghash[${flagname}]=${flagvalue}
        elif [ -n "${words[ $((c+1)) ]}" ] ; then
            flaghash[${flagname}]=${words[ $((c+1)) ]}
        else
            flaghash[${flagname}]="true" # pad "true" for bool flag
        fi
    fi

    # skip the argument to a two word flag
    if [[ ${words[c]} != *"="* ]] && __etcdctl_contains_word "${words[c]}" "${two_word_flags[@]}"; then
        __etcdctl_debug "${FUNCNAME[0]}: found a flag ${words[c]}, skip the next argument"
        c=$((c+1))
        # if we are looking for a flags value, don't show commands
        if [[ $c -eq $cword ]]; then
            commands=()
        fi
    fi

    c=$((c+1))

}

__etcdctl_handle_noun()
{
    __etcdctl_debug "${FUNCNAME[0]}: c is $c words[c] is ${words[c]}"

    if __etcdctl_contains_word "${words[c]}" "${must_have_one_noun[@]}"; then
        must_have_one_noun=()
    elif __etcdctl_contains_word "${words[c]}" "${noun_aliases[@]}"; then
        must_have_one_noun=()
    fi

    nouns+=("${words[c]}")
    c=$((c+1))
}

__etcdctl_handle_command()
{
    __etcdctl_debug "${FUNCNAME[0]}: c is $c words[c] is ${words[c]}"

    local next_command
    if [[ -n ${last_command} ]]; then
        next_command="_${last_command}_${words[c]//:/__}"
    else
        if [[ $c -eq 0 ]]; then
            next_command="_etcdctl_root_command"
        else
            next_command="_${words[c]//:/__}"
        fi
    fi
    c=$((c+1))
    __etcdctl_debug "${FUNCNAME[0]}: looking for ${next_command}"
    declare -F "$next_command" >/dev/null && $next_command
}

__etcdctl_handle_word()
{
    if [[ $c -ge $cword ]]; then
        __etcdctl_handle_reply
        return
    fi
    __etcdctl_debug "${FUNCNAME[0]}: c is $c words[c] is ${words[c]}"
    if [[ "${words[c]}" == -* ]]; then
        __etcdctl_handle_flag
    elif __etcdctl_contains_word "${words[c]}" "${commands[@]}"; then
        __etcdctl_handle_command
    elif [[ $c -eq 0 ]]; then
        __etcdctl_handle_command
    elif __etcdctl_contains_word "${words[c]}" "${command_aliases[@]}"; then
        # aliashash variable is an associative array which is only supported in bash > 3.
        if [[ -z "${BASH_VERSION:-}" || "${BASH_VERSINFO[0]:-}" -gt 3 ]]; then
            words[c]=${aliashash[${words[c]}]}
            __etcdctl_handle_command
        else
            __etcdctl_handle_noun
        fi
    else
        __etcdctl_handle_noun
    fi
    __etcdctl_handle_word
}

_etcdctl_alarm_disarm()
{
    last_command="etcdctl_alarm_disarm"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_alarm_list()
{
    last_command="etcdctl_alarm_list"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_alarm()
{
    last_command="etcdctl_alarm"

    command_aliases=()

    commands=()
    commands+=("disarm")
    commands+=("list")

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_auth_disable()
{
    last_command="etcdctl_auth_disable"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_auth_enable()
{
    last_command="etcdctl_auth_enable"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_auth_status()
{
    last_command="etcdctl_auth_status"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_auth()
{
    last_command="etcdctl_auth"

    command_aliases=()

    commands=()
    commands+=("disable")
    commands+=("enable")
    commands+=("status")

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_check_datascale()
{
    last_command="etcdctl_check_datascale"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--auto-compact")
    local_nonpersistent_flags+=("--auto-compact")
    flags+=("--auto-defrag")
    local_nonpersistent_flags+=("--auto-defrag")
    flags+=("--load=")
    two_word_flags+=("--load")
    local_nonpersistent_flags+=("--load")
    local_nonpersistent_flags+=("--load=")
    flags+=("--prefix=")
    two_word_flags+=("--prefix")
    local_nonpersistent_flags+=("--prefix")
    local_nonpersistent_flags+=("--prefix=")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_check_perf()
{
    last_command="etcdctl_check_perf"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--auto-compact")
    local_nonpersistent_flags+=("--auto-compact")
    flags+=("--auto-defrag")
    local_nonpersistent_flags+=("--auto-defrag")
    flags+=("--load=")
    two_word_flags+=("--load")
    flags_with_completion+=("--load")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    local_nonpersistent_flags+=("--load")
    local_nonpersistent_flags+=("--load=")
    flags+=("--prefix=")
    two_word_flags+=("--prefix")
    local_nonpersistent_flags+=("--prefix")
    local_nonpersistent_flags+=("--prefix=")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_check()
{
    last_command="etcdctl_check"

    command_aliases=()

    commands=()
    commands+=("datascale")
    commands+=("perf")

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_compaction()
{
    last_command="etcdctl_compaction"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--physical")
    local_nonpersistent_flags+=("--physical")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_completion()
{
    last_command="etcdctl_completion"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--help")
    flags+=("-h")
    local_nonpersistent_flags+=("--help")
    local_nonpersistent_flags+=("-h")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    must_have_one_noun+=("bash")
    must_have_one_noun+=("fish")
    must_have_one_noun+=("powershell")
    must_have_one_noun+=("zsh")
    noun_aliases=()
}

_etcdctl_defrag()
{
    last_command="etcdctl_defrag"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cluster")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_del()
{
    last_command="etcdctl_del"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--from-key")
    local_nonpersistent_flags+=("--from-key")
    flags+=("--prefix")
    local_nonpersistent_flags+=("--prefix")
    flags+=("--prev-kv")
    local_nonpersistent_flags+=("--prev-kv")
    flags+=("--range")
    local_nonpersistent_flags+=("--range")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_downgrade_cancel()
{
    last_command="etcdctl_downgrade_cancel"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_downgrade_enable()
{
    last_command="etcdctl_downgrade_enable"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_downgrade_validate()
{
    last_command="etcdctl_downgrade_validate"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_downgrade()
{
    last_command="etcdctl_downgrade"

    command_aliases=()

    commands=()
    commands+=("cancel")
    commands+=("enable")
    commands+=("validate")

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_elect()
{
    last_command="etcdctl_elect"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--listen")
    flags+=("-l")
    local_nonpersistent_flags+=("--listen")
    local_nonpersistent_flags+=("-l")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_endpoint_hashkv()
{
    last_command="etcdctl_endpoint_hashkv"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--rev=")
    two_word_flags+=("--rev")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--cluster")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_endpoint_health()
{
    last_command="etcdctl_endpoint_health"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--cluster")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_endpoint_status()
{
    last_command="etcdctl_endpoint_status"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--cluster")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_endpoint()
{
    last_command="etcdctl_endpoint"

    command_aliases=()

    commands=()
    commands+=("hashkv")
    commands+=("health")
    commands+=("status")

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cluster")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_get()
{
    last_command="etcdctl_get"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--consistency=")
    two_word_flags+=("--consistency")
    flags_with_completion+=("--consistency")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    local_nonpersistent_flags+=("--consistency")
    local_nonpersistent_flags+=("--consistency=")
    flags+=("--count-only")
    local_nonpersistent_flags+=("--count-only")
    flags+=("--from-key")
    local_nonpersistent_flags+=("--from-key")
    flags+=("--keys-only")
    local_nonpersistent_flags+=("--keys-only")
    flags+=("--limit=")
    two_word_flags+=("--limit")
    local_nonpersistent_flags+=("--limit")
    local_nonpersistent_flags+=("--limit=")
    flags+=("--max-create-rev=")
    two_word_flags+=("--max-create-rev")
    local_nonpersistent_flags+=("--max-create-rev")
    local_nonpersistent_flags+=("--max-create-rev=")
    flags+=("--max-mod-rev=")
    two_word_flags+=("--max-mod-rev")
    local_nonpersistent_flags+=("--max-mod-rev")
    local_nonpersistent_flags+=("--max-mod-rev=")
    flags+=("--min-create-rev=")
    two_word_flags+=("--min-create-rev")
    local_nonpersistent_flags+=("--min-create-rev")
    local_nonpersistent_flags+=("--min-create-rev=")
    flags+=("--min-mod-rev=")
    two_word_flags+=("--min-mod-rev")
    local_nonpersistent_flags+=("--min-mod-rev")
    local_nonpersistent_flags+=("--min-mod-rev=")
    flags+=("--order=")
    two_word_flags+=("--order")
    flags_with_completion+=("--order")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    local_nonpersistent_flags+=("--order")
    local_nonpersistent_flags+=("--order=")
    flags+=("--prefix")
    local_nonpersistent_flags+=("--prefix")
    flags+=("--print-value-only")
    local_nonpersistent_flags+=("--print-value-only")
    flags+=("--rev=")
    two_word_flags+=("--rev")
    local_nonpersistent_flags+=("--rev")
    local_nonpersistent_flags+=("--rev=")
    flags+=("--sort-by=")
    two_word_flags+=("--sort-by")
    flags_with_completion+=("--sort-by")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    local_nonpersistent_flags+=("--sort-by")
    local_nonpersistent_flags+=("--sort-by=")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_help()
{
    last_command="etcdctl_help"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    has_completion_function=1
    noun_aliases=()
}

_etcdctl_lease_grant()
{
    last_command="etcdctl_lease_grant"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_lease_keep-alive()
{
    last_command="etcdctl_lease_keep-alive"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--once")
    local_nonpersistent_flags+=("--once")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_lease_list()
{
    last_command="etcdctl_lease_list"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_lease_revoke()
{
    last_command="etcdctl_lease_revoke"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_lease_timetolive()
{
    last_command="etcdctl_lease_timetolive"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--keys")
    local_nonpersistent_flags+=("--keys")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_lease()
{
    last_command="etcdctl_lease"

    command_aliases=()

    commands=()
    commands+=("grant")
    commands+=("keep-alive")
    commands+=("list")
    commands+=("revoke")
    commands+=("timetolive")

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_lock()
{
    last_command="etcdctl_lock"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--ttl=")
    two_word_flags+=("--ttl")
    local_nonpersistent_flags+=("--ttl")
    local_nonpersistent_flags+=("--ttl=")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_make-mirror()
{
    last_command="etcdctl_make-mirror"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--dest-cacert=")
    two_word_flags+=("--dest-cacert")
    local_nonpersistent_flags+=("--dest-cacert")
    local_nonpersistent_flags+=("--dest-cacert=")
    flags+=("--dest-cert=")
    two_word_flags+=("--dest-cert")
    local_nonpersistent_flags+=("--dest-cert")
    local_nonpersistent_flags+=("--dest-cert=")
    flags+=("--dest-insecure-transport")
    local_nonpersistent_flags+=("--dest-insecure-transport")
    flags+=("--dest-key=")
    two_word_flags+=("--dest-key")
    local_nonpersistent_flags+=("--dest-key")
    local_nonpersistent_flags+=("--dest-key=")
    flags+= "**********"=")
    two_word_flags+= "**********"
    local_nonpersistent_flags+= "**********"
    local_nonpersistent_flags+= "**********"=")
    flags+=("--dest-prefix=")
    two_word_flags+=("--dest-prefix")
    local_nonpersistent_flags+=("--dest-prefix")
    local_nonpersistent_flags+=("--dest-prefix=")
    flags+=("--dest-user=")
    two_word_flags+=("--dest-user")
    local_nonpersistent_flags+=("--dest-user")
    local_nonpersistent_flags+=("--dest-user=")
    flags+=("--max-txn-ops=")
    two_word_flags+=("--max-txn-ops")
    local_nonpersistent_flags+=("--max-txn-ops")
    local_nonpersistent_flags+=("--max-txn-ops=")
    flags+=("--no-dest-prefix")
    local_nonpersistent_flags+=("--no-dest-prefix")
    flags+=("--prefix=")
    two_word_flags+=("--prefix")
    local_nonpersistent_flags+=("--prefix")
    local_nonpersistent_flags+=("--prefix=")
    flags+=("--rev=")
    two_word_flags+=("--rev")
    local_nonpersistent_flags+=("--rev")
    local_nonpersistent_flags+=("--rev=")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_member_add()
{
    last_command="etcdctl_member_add"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--learner")
    local_nonpersistent_flags+=("--learner")
    flags+=("--peer-urls=")
    two_word_flags+=("--peer-urls")
    local_nonpersistent_flags+=("--peer-urls")
    local_nonpersistent_flags+=("--peer-urls=")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_member_list()
{
    last_command="etcdctl_member_list"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--consistency=")
    two_word_flags+=("--consistency")
    local_nonpersistent_flags+=("--consistency")
    local_nonpersistent_flags+=("--consistency=")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_member_promote()
{
    last_command="etcdctl_member_promote"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_member_remove()
{
    last_command="etcdctl_member_remove"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_member_update()
{
    last_command="etcdctl_member_update"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--peer-urls=")
    two_word_flags+=("--peer-urls")
    local_nonpersistent_flags+=("--peer-urls")
    local_nonpersistent_flags+=("--peer-urls=")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_member()
{
    last_command="etcdctl_member"

    command_aliases=()

    commands=()
    commands+=("add")
    commands+=("list")
    commands+=("promote")
    commands+=("remove")
    commands+=("update")

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_move-leader()
{
    last_command="etcdctl_move-leader"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_put()
{
    last_command="etcdctl_put"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--ignore-lease")
    local_nonpersistent_flags+=("--ignore-lease")
    flags+=("--ignore-value")
    local_nonpersistent_flags+=("--ignore-value")
    flags+=("--lease=")
    two_word_flags+=("--lease")
    local_nonpersistent_flags+=("--lease")
    local_nonpersistent_flags+=("--lease=")
    flags+=("--prev-kv")
    local_nonpersistent_flags+=("--prev-kv")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_role_add()
{
    last_command="etcdctl_role_add"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_role_delete()
{
    last_command="etcdctl_role_delete"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_role_get()
{
    last_command="etcdctl_role_get"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_role_grant-permission()
{
    last_command="etcdctl_role_grant-permission"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--from-key")
    local_nonpersistent_flags+=("--from-key")
    flags+=("--prefix")
    local_nonpersistent_flags+=("--prefix")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_role_list()
{
    last_command="etcdctl_role_list"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_role_revoke-permission()
{
    last_command="etcdctl_role_revoke-permission"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--from-key")
    local_nonpersistent_flags+=("--from-key")
    flags+=("--prefix")
    local_nonpersistent_flags+=("--prefix")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_role()
{
    last_command="etcdctl_role"

    command_aliases=()

    commands=()
    commands+=("add")
    commands+=("delete")
    commands+=("get")
    commands+=("grant-permission")
    commands+=("list")
    commands+=("revoke-permission")

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_snapshot_save()
{
    last_command="etcdctl_snapshot_save"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_snapshot()
{
    last_command="etcdctl_snapshot"

    command_aliases=()

    commands=()
    commands+=("save")

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_txn()
{
    last_command="etcdctl_txn"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--interactive")
    flags+=("-i")
    local_nonpersistent_flags+=("--interactive")
    local_nonpersistent_flags+=("-i")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_user_add()
{
    last_command="etcdctl_user_add"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--interactive")
    local_nonpersistent_flags+=("--interactive")
    flags+= "**********"=")
    two_word_flags+= "**********"
    local_nonpersistent_flags+= "**********"
    local_nonpersistent_flags+= "**********"=")
    flags+= "**********"
    local_nonpersistent_flags+= "**********"
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_user_delete()
{
    last_command="etcdctl_user_delete"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_user_get()
{
    last_command="etcdctl_user_get"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--detail")
    local_nonpersistent_flags+=("--detail")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_user_grant-role()
{
    last_command="etcdctl_user_grant-role"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_user_list()
{
    last_command="etcdctl_user_list"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_user_passwd()
{
    last_command="etcdctl_user_passwd"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--interactive")
    local_nonpersistent_flags+=("--interactive")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_user_revoke-role()
{
    last_command="etcdctl_user_revoke-role"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_user()
{
    last_command="etcdctl_user"

    command_aliases=()

    commands=()
    commands+=("add")
    commands+=("delete")
    commands+=("get")
    commands+=("grant-role")
    commands+=("list")
    commands+=("passwd")
    commands+=("revoke-role")

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_version()
{
    last_command="etcdctl_version"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_watch()
{
    last_command="etcdctl_watch"

    command_aliases=()

    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--interactive")
    flags+=("-i")
    local_nonpersistent_flags+=("--interactive")
    local_nonpersistent_flags+=("-i")
    flags+=("--prefix")
    local_nonpersistent_flags+=("--prefix")
    flags+=("--prev-kv")
    local_nonpersistent_flags+=("--prev-kv")
    flags+=("--progress-notify")
    local_nonpersistent_flags+=("--progress-notify")
    flags+=("--rev=")
    two_word_flags+=("--rev")
    local_nonpersistent_flags+=("--rev")
    local_nonpersistent_flags+=("--rev=")
    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_etcdctl_root_command()
{
    last_command="etcdctl"

    command_aliases=()

    commands=()
    commands+=("alarm")
    commands+=("auth")
    commands+=("check")
    commands+=("compaction")
    commands+=("completion")
    commands+=("defrag")
    commands+=("del")
    commands+=("downgrade")
    commands+=("elect")
    commands+=("endpoint")
    commands+=("get")
    commands+=("help")
    commands+=("lease")
    commands+=("lock")
    commands+=("make-mirror")
    commands+=("member")
    commands+=("move-leader")
    commands+=("put")
    commands+=("role")
    commands+=("snapshot")
    commands+=("txn")
    commands+=("user")
    commands+=("version")
    commands+=("watch")

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cacert=")
    two_word_flags+=("--cacert")
    flags+=("--cert=")
    two_word_flags+=("--cert")
    flags+=("--command-timeout=")
    two_word_flags+=("--command-timeout")
    flags+=("--debug")
    flags+=("--dial-timeout=")
    two_word_flags+=("--dial-timeout")
    flags+=("--discovery-srv=")
    two_word_flags+=("--discovery-srv")
    two_word_flags+=("-d")
    flags+=("--discovery-srv-name=")
    two_word_flags+=("--discovery-srv-name")
    flags+=("--endpoints=")
    two_word_flags+=("--endpoints")
    flags+=("--hex")
    flags+=("--insecure-discovery")
    flags+=("--insecure-skip-tls-verify")
    flags+=("--insecure-transport")
    flags+=("--keepalive-time=")
    two_word_flags+=("--keepalive-time")
    flags+=("--keepalive-timeout=")
    two_word_flags+=("--keepalive-timeout")
    flags+=("--key=")
    two_word_flags+=("--key")
    flags+=("--max-recv-bytes=")
    two_word_flags+=("--max-recv-bytes")
    flags+=("--max-request-bytes=")
    two_word_flags+=("--max-request-bytes")
    flags+= "**********"=")
    two_word_flags+= "**********"
    flags+=("--user=")
    two_word_flags+=("--user")
    flags+=("--write-out=")
    two_word_flags+=("--write-out")
    flags_with_completion+=("--write-out")
    flags_completion+=("__etcdctl_handle_go_custom_completion")
    two_word_flags+=("-w")
    flags_with_completion+=("-w")
    flags_completion+=("__etcdctl_handle_go_custom_completion")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

__start_etcdctl()
{
    local cur prev words cword split
    declare -A flaghash 2>/dev/null || :
    declare -A aliashash 2>/dev/null || :
    if declare -F _init_completion >/dev/null 2>&1; then
        _init_completion -s || return
    else
        __etcdctl_init_completion -n "=" || return
    fi

    local c=0
    local flag_parsing_disabled=
    local flags=()
    local two_word_flags=()
    local local_nonpersistent_flags=()
    local flags_with_completion=()
    local flags_completion=()
    local commands=("etcdctl")
    local command_aliases=()
    local must_have_one_flag=()
    local must_have_one_noun=()
    local has_completion_function=""
    local last_command=""
    local nouns=()
    local noun_aliases=()

    __etcdctl_handle_word
}

if [[ $(type -t compopt) = "builtin" ]]; then
    complete -o default -F __start_etcdctl etcdctl
else
    complete -o default -o nospace -F __start_etcdctl etcdctl
fi

# ex: ts=4 sw=4 et filetype=sh
e -F __start_etcdctl etcdctl
fi

# ex: ts=4 sw=4 et filetype=sh
