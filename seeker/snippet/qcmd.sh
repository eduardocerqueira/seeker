#date: 2024-02-01T17:05:00Z
#url: https://api.github.com/gists/47012dd1d9bdc5b31683343671d78c92
#owner: https://api.github.com/users/elnikkis

#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: [Q=QueueName] [P=NProcs] [T=NThreads] [C=NCpus] [M=Memory (GB)] [W=WallTime (hour)] [J=JobName] [O=LogDir] bash $0 [-I|command]"
    exit 1
fi


# Set default values
R="p=${P:=1}:t=${T:=1}:c=${C:=1}:m=${M:=3}G"
if [ -z "$W" ]; then
    W=1
fi
if [ -z "$J" ]; then
    J="qcmd"
fi
if [ -n "$Q" ]; then
    partition="-p $Q"
fi
if [ -z "$O" ]; then
    O="."
fi

# Run
if [ "$*" = "-I" ]; then
    tssrun $partition --rsc "$R" -t "${W}:00" -pty /bin/bash
else
    export QCMD_CMD=$*
    sbatch $partition -t "${W}:00:00" --rsc "$R" -J "$J" -o "${O}/%x.o%j.out" << EOM
#!/bin/bash
set -x
hostname
date
srun bash -c "$QCMD_CMD"
date
EOM
fi
