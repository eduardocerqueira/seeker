#date: 2022-10-17T17:28:27Z
#url: https://api.github.com/gists/b7d6daf394d3b7fca73a2e5b5f30c7f4
#owner: https://api.github.com/users/danbruegge

#!/bin/sh

NOW=$(date +"%Y-%m-%d-%H-%M-%S")
COMMAND='pnpm cypress run --spec file.spec.js';
OUTPUT_PATH="output/${NOW}";
OUTPUT_FILENAME="${OUTPUT_PATH}/  run";
RUN_COUNTER=1;
RUN_LOOP=true;

beep() {
    (speaker-test -t sine -f 1000) & pid=$!;
    sleep 0.3s;
    kill -9 $pid;
}

mkdir -p -- "$OUTPUT_PATH" > /dev/null

echo "🚀 Start script - ⏱  ${NOW}";

while $RUN_LOOP; do
    OUTPUT_FILE="${OUTPUT_FILENAME}_${RUN_COUNTER}"

    if test -f "$OUTPUT_FILE"; then
        if grep -i -e "failing" -e "error" -e "cy:command ✘" "$OUTPUT_FILE"; then
            echo "🛑 Found Error...Stopping Script";
            beep > /dev/null;
            RUN_LOOP=false;
        fi
    fi
    
    echo "💨 Run: ${RUN_COUNTER}";

    $COMMAND >> "$OUTPUT_FILE"

    RUN_COUNTER=$((RUN_COUNTER+1))
done

echo "🏁 End script   - ⏱  $(date +"%Y-%m-%d-%H-%M-%S")";