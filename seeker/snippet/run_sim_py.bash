#date: 2025-02-25T17:00:38Z
#url: https://api.github.com/gists/d9fbe9ce1d9c3649756ec7cd22f910f2
#owner: https://api.github.com/users/tam17aki

#!/bin/bash

CURDIR=$(cd $(dirname $0);pwd)
LOGDIR=${CURDIR}/log

mkdir -p ${LOGDIR}

HOPNET=hopnet_dynamics.py

N_NEURONS=1000   # The number of neurons
N_PATTERNS=80   # The maximum number of patterns
N_STEPS=25
SELF_CONNECT=False

for i in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9; do
    LOG_FILE=log_${i}.txt
    echo "similarity ${i}"
    python3.11 ${CURDIR}/${HOPNET} --num_neurons ${N_NEURONS} \
               --num_patterns ${N_PATTERNS} \
               --num_steps ${N_STEPS} \
               --similarity ${i} \
               --self_connection ${SELF_CONNECT} \
               --log_file ${LOGDIR}/${LOG_FILE}  
done
