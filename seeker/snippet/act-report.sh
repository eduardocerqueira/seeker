#date: 2025-04-02T17:03:11Z
#url: https://api.github.com/gists/c973cd905460150706a16b11075fab6a
#owner: https://api.github.com/users/gastmaier

#!/bin/bash

# Pipes ACT very verbose output and stdout only GitHub Actions annotations (warnings and errors)
# The full log is saved to runner.log
# Usage:
#   source act_report.sh
#   act_report act -W .github/workflows/kernel.yml --remote-name=public
act_report() {
        local err=0
        declare -A ci_job_index
        local nindex=0

        $@ | (while IFS= read -r row; do
                if [[ "$row" =~ ^\[([^]]+)\][[:space:]]+([[:alnum:]_[:punct:]])[[:space:]]+::(warning|error)[[:space:]]+(.*)::(.*)$ ]]; then
                        ci_job="${BASH_REMATCH[1]}"
                        emoji="${BASH_REMATCH[2]}"
                        level="${BASH_REMATCH[3]}"
                        file_info="${BASH_REMATCH[4]}"
                        message="${BASH_REMATCH[5]}"
                        message=$(echo "$message" | sed 's/%0A/\n       /g')

                        if [[ "$level" == "error" ]]; then
                                level="\e[31merror\e[0m  "
                        else
                                level="\e[33mwarning\e[0m"
                        fi

                        if [[ ! -v ci_job_index["$ci_job"] ]]; then
                                ci_job_index["$ci_job"]="$nindex"
                                ((nindex++))
                        fi
                        ci_index=${ci_job_index["$ci_job"]}
                        ci_job="\e[3$(( 1 + $ci_index ))m[$ci_job]\e[0m"
                        file_info=$(echo "$file_info" | sed -E 's/file=([^,]+),line=([0-9]+)(,(.*))?/\1:\2 \4/g')

                        printf "$ci_job $level $file_info\n\n" | tee -a runner.log
                        printf "\t$message\n" | tee -a runner.log
                elif [[ "$row" =~ ^\[([^]]+)\][[:space:]]+([[:alnum:]_[:punct:]])[[:space:]]+Run[[:space:]](.*)$ ]]; then
                        ci_job="${BASH_REMATCH[1]}"
                        emoji="${BASH_REMATCH[2]}"
                        step="${BASH_REMATCH[3]}"

                        if [[ ! -v ci_job_index["$ci_job"] ]]; then
                                ci_job_index["$ci_job"]="$nindex"
                                ((nindex++))
                        fi
                        ci_index=${ci_job_index["$ci_job"]}
                        ci_job="\e[3$(( 1 + $ci_index ))m[$ci_job]\e[0m"
                        step="$step"
                        printf "$ci_job $step\n" | tee -a runner.log
                else
                        echo $row >> runner.log
                fi
        done) ; err=${PIPESTATUS[0]}

        return $err
}