#date: 2023-12-07T16:55:01Z
#url: https://api.github.com/gists/b254b4c98aafd739b9b809171212a407
#owner: https://api.github.com/users/mpatou

#!/bin/bash
# from
# https://gist.githubusercontent.com/zish/85dccece461e050077997ff5d7d9c9d4/raw/c4719a2443e4be0010fd4c3dbc8a94ad14b4e6ee/terraform_bash_completion.sh
# Bash Terraform completion
#
# Originally adapted from
# https://gist.github.com/cornfeedhobo/8bc08747ec3add1fc5adb2edb7cd68d3
#
# Author: Jeremy Melanson
#
# Features of this version:
# - Uses built-in bash routines for text processing, instead of external tools
#   (awk, sed, grep, ...).
#
# - fixes the retrieval of options from the Terraform executble.
#
# - Optional _init_terraform_completion function, which can enable
#   command-completion for multiple Terraform executables.
#

#--- Get options listing from Terraform command.
#
_terraform_completion_get_opts () {
  local CMD_EXEC="${1}"
  shift
  local TF_OPT="${1}"
  shift
  local words=("${@}")

  local IFS=$'\n'
  local full_command=0
  for O in $(${CMD_EXEC} -help); do
    if [[ "${O}" =~ ^\ +([^\= ]+) ]]; then
      if [[ "${BASH_REMATCH[1]}" == "${TF_OPT}" ]]; then
        full_command=1
        break
      fi
    fi
  done
  #-- "terraform -help"
  if [[ "${TF_OPT}" == "_" || "${TF_OPT}" == "" ]] || [[ "$full_command" -eq 0 ]]; then

    for O in $(${CMD_EXEC} -help); do
      if [[ "${O}" =~ ^\ +([^\= ]+) ]]; then
        skip=0
        for w in "${words[@]}"; do
          if [[ "${w}" == "${BASH_REMATCH[1]}" ]]; then
            skip=1
            break
          fi
        done
        if [[ ${skip} -eq 0 ]]; then
          echo -e "${BASH_REMATCH[1]}"
        fi
      fi
    done

  #-- "terraform -help XXXX"
  else

    for O in $(${CMD_EXEC} -help "${TF_OPT}"); do
      if [[ "${O}" =~ ^\ +(-[^\ =]+=?) ]]; then
        echo -e "${BASH_REMATCH[1]}"
      fi
    done

  fi
}


#--- This function is passed to 'complete' for handling completion.
#
_terraform_completion () {
  local cur prev words cword opts

  local O_COMP_WORDBREAKS="${COMP_WORDBREAKS}"
  export COMP_WORDBREAKS="${COMP_WORDBREAKS//=/}"

  _init_completion -s || return

  _get_comp_words_by_ref -n : cur prev words cword
  COMPREPLY=()

  local opts=""

  local CMD_EXEC="${COMP_WORDS[0]}"

  if [[ ${cur} == "=" ]]; then 
    cur=""
  fi
  if [[ ${cword} -eq 1 ]]; then
    if [[ ${cur} == -* ]]; then
      if [[ ${cur} == -c* ]]; then
        # Do not append a space to the current completion
        compopt -o nospace
        # Remove -chdir from $cur to get the current directory specified
        local CUR_DIR=${cur//-chdir=/}

        # This is almost 100% not working on bash because of the spliting after the =
        # Add prefix (-P) `-chdir` to all the completions
        opts=$(compgen -P "-chdir=" -d -- "${CUR_DIR}")
        opts="${opts:--chdir=}"

      else
        opts="-chdir= -help -version"

      fi
    else
      opts=$(_terraform_completion_get_opts "${CMD_EXEC}")

    fi

  elif [[ ${cword} -gt 1 ]]; then
    first="${words[0]}"
    second="${words[1]}"
    if [[ ${cword} -lt 4 ]] && [[ ${first} == "-chdir" || ${second} == '-chdir' ]] ; then
      # Do not append a space to the current completion
      compopt -o nospace
      # Remove -chdir from $prev to get the current directory specified
      local CUR_DIR=${cur//=/}
      # Add prefix (-P) `-chdir` to all the completions
      opts=$(compgen  -d -- "${CUR_DIR}"|tr "\n" " ")
      if [[ "${CUR_DIR}" == "" ]]; then
        opts=". .. $opts"
      fi
    elif [[ ${cword} -eq 2 && ${prev} == -help ]]; then
      opts=$(_terraform_completion_get_opts "${CMD_EXEC}")
    else
      if [[ "${NEW_OPT_TYPE}" != "" ]]; then
        compopt -o nospace
      else
        # start with 1 to skip the first element that is the name of the command
        dash_or_similar=1
        local TF_COMMAND="_"
        for w in "${words[@]}"; do

          if [[ ${w} == -* ]]; then
            dash_or_similar=1
          elif [[ ${w} == '=' && ${dash_or_similar} -eq 1 ]]; then
            dash_or_similar=1
          elif [[ ${dash_or_similar} -eq 1 ]]; then
            dash_or_similar=0
          elif [[ ${dash_or_similar} -eq 0 && "${w}" != ""  ]]; then
            TF_COMMAND="${w}"
            break
          fi
        done
        opts=$(_terraform_completion_get_opts "${CMD_EXEC}" "${TF_COMMAND}" "${words[@]}")

      fi
    fi
  fi
  IFS=', ' read -r -a COMPREPLY<<< $(compgen -W "${opts}" -- "${cur}" | tr "\n" " ")

  if [[ ${#COMPREPLY[@]} -eq 1 ]]; then
    if [[ "${COMPREPLY[0]}" =~ =$ ]]; then
      compopt -o nospace
    fi
  fi

  export COMP_WORDBREAKS="${O_COMP_WORDBREAKS}"

  return 0
}


#--- Initialize Bash Command Completion for multiple Terraform executables.
# It searches the PATH for files starting with "terraform", but does not
# contain "-" characters. This avoids adding Completion for third-party
# Terraform provider plugins that exist as a separate executable
# (terraform-provider-XXXX).
#
_init_terraform_completion () {

  #-- Regex used when looking for terraform executables.
  # Looks for "terraform", or "terraform[anything else that isn't a dash]".
  # This enables Command Completion for multiple versions of Terraform,
  # if you have them.
  #
  local TF_EXEC_PREFIX='^terrafor(m[^-]+|m)$'
  local ORIG_DIR="${PWD}"

  local IFS=':'
  for P in ${PATH}; do
    if [ -d "${P}" ]; then
      cd "${P}" || return "255"

    else
      continue

    fi

    for E in *; do
      if [[ "${E}" =~ ${TF_EXEC_PREFIX} ]]; then
        complete -F _terraform_completion "${E}"
      fi
    done

  done

  cd "${ORIG_DIR}" || return "255"
}

complete -F _terraform_completion terraform


#--- Optionally enable command completion for multiple Terraform executables.
# It currently works with executables in the PATH, that are named similar
# to "terraform_XXXX".
#
# ** If your files are named differently, then you may need to modify the REGEX
# ** in TF_EXEC_PREFIX to suit your needs.
#
# This provides a little simplicity, when working with multiple Terraform versions.
#
# Uncomment this line to enable:
#
_init_terraform_completion

#--- Remove the initialization function. It is only needed once.
unset -f _init_terraform_completion


#--- Include command completion for the optional "tf" command.
complete -F _terraform_completion tf
