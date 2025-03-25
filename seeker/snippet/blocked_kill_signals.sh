#date: 2025-03-25T16:58:21Z
#url: https://api.github.com/gists/84e063dc0286a8173a11dc8ae975f150
#owner: https://api.github.com/users/jrelo

#!/bin/bash

function blocked_signals {
    # convert the hexadecimal input to a binary string, ensuring uppercase
    local hex_mask=$(echo "$1" | tr '[:lower:]' '[:upper:]')
    local binary_mask=$(echo "obase=2; ibase=16; $hex_mask" | bc)

    # calculate the number of bits (signals) to process, typically 64 for Linux
    local num_bits=64
    local padded_binary=$(printf "%0${num_bits}s" $binary_mask | sed 's/ /0/g')

    # reverse the binary string to align with signal numbers
    local reversed_binary=$(echo $padded_binary | rev)

    local blocked_signals=()

    # iterate over each bit in the reversed binary string
    local length=${#reversed_binary}
    for (( i=0; i<$length; i++ )); do
        if [ "${reversed_binary:$i:1}" == "1" ]; then
            # bit position corresponds to the signal number
            local signal_number=$((i + 1))
            # get signal name from the number using `kill -l`
            local signal_name=$(kill -l $signal_number 2>/dev/null)
            # append the signal name to the list if its valid
            if [ $? -eq 0 ]; then
                blocked_signals+=("$signal_name")
            fi
        fi
    done

    echo "Blocked signals by bitmask $hex_mask ($binary_mask):"
    if [ ${#blocked_signals[@]} -gt 0 ]; then
        echo "${blocked_signals[@]}"
    else
        echo "No signals are blocked."
    fi
}

# Example usage:
# └─# grep SigBlk /proc/993/status
# SigBlk: ffffffffe7ffb9eff
# └─# show_blocked_signals "ffffffffe7ffb9eff"
# Blocked signals by bitmask FFFFFFFFE7FFB9EFF (11111111111111111111111111111111111001111111111110111001111011111111):
# HUP INT QUIT ILL TRAP ABRT BUS FPE USR1 SEGV USR2 PIPE STKFLT CHLD CONT TSTP TTIN TTOU URG XCPU XFSZ VTALRM PROF WINCH IO PWR SYS RTMIN RTMIN+1 RTMIN+2 RTMIN+3 RTMIN+4 RTMIN+5 RTMIN+6 RTMIN+7 RTMIN+8 RTMIN+9 RTMIN+10 RTMIN+11 RTMIN+12 RTMIN+13 RTMIN+14 RTMIN+15 RTMAX-14 RTMAX-13 RTMAX-12 RTMAX-11 RTMAX-10 RTMAX-9 RTMAX-8 RTMAX-7 RTMAX-6 RTMAX-5 RTMAX-4 RTMAX-3 RTMAX-2 RTMAX-1 RTMAX 
