#date: 2025-10-30T17:05:34Z
#url: https://api.github.com/gists/65598df774ce59b2d5256c72e8ddf59b
#owner: https://api.github.com/users/Muhammad-Raiyan

#!/bin/bash
# Usage: ./run_all_tests.sh <BrokerName> <ExecutionGroupName> <QueueManagerName> [--log-mode console|file]

# -----------------------------
# Parameters
# -----------------------------
BROKER_NAME=$1
EG_NAME=$2
QM_NAME=$3
LOG_MODE="file"

# Simplified optional flag parsing
if [ "$4" = "--log-mode" ]; then
    LOG_MODE="$5"
fi

# Base directories
SERVER_DIR="/var/mqsi/components/${BROKER_NAME}/servers/${EG_NAME}"
RUN_DIR="${SERVER_DIR}/run"
TIMESTAMP=$(date +%F_%H%M%S)
LOG_BASE="/tmp/test_run_${TIMESTAMP}/${BROKER_NAME}/${EG_NAME}"
SUMMARY_FILE="/tmp/test_run_${TIMESTAMP}/summary.csv"

mkdir -p "$LOG_BASE"

# Validation
if [ ! -d "$SERVER_DIR" ]; then
    echo "Error: Server directory not found: $SERVER_DIR"
    exit 1
fi
if [ ! -d "$RUN_DIR" ]; then
    echo "Error: Run directory not found: $RUN_DIR"
    exit 1
fi

echo "-------------------------------------------------------"
echo "Running tests for:"
echo " Broker          : $BROKER_NAME"
echo " Execution Group : $EG_NAME"
echo " Queue Manager   : $QM_NAME"
echo " Log Mode        : $LOG_MODE"
echo " Log Directory   : $LOG_BASE"
echo " Summary File    : $SUMMARY_FILE"
echo "-------------------------------------------------------"

# Initialize summary file
echo "Test Project,Status,Exit Code,Log File" > "$SUMMARY_FILE"

OVERALL_EXIT=0

for TEST_PROJECT in "$RUN_DIR"/GenTest_*; do
    if [ -d "$TEST_PROJECT" ]; then
        PROJECT_NAME=$(basename "$TEST_PROJECT")
        LOG_FILE="${LOG_BASE}/${PROJECT_NAME}.log"
        CMD="IntegrationServer --work-dir ${SERVER_DIR} --no-nodejs --start-msgflows false --mq-queue-manager-name ${QM_NAME} --test-project ${PROJECT_NAME}"

        echo "Running test: $PROJECT_NAME"

        if [ "$LOG_MODE" = "console" ]; then
            echo "-------------------------------------------------------"
            echo "Command: $CMD"
            echo "-------------------------------------------------------"
            $CMD
            EXIT_CODE=$?
        else
            echo "  Writing output to: $LOG_FILE"
            echo "-------------------------------------------------------" > "$LOG_FILE"
            echo "Command: $CMD" >> "$LOG_FILE"
            echo "Started at: $(date)" >> "$LOG_FILE"
            echo "-------------------------------------------------------" >> "$LOG_FILE"
            $CMD >>"$LOG_FILE" 2>&1
            EXIT_CODE=$?
        fi

        if [ $EXIT_CODE -ne 0 ]; then
            STATUS="FAIL"
            OVERALL_EXIT=1
        else
            STATUS="PASS"
        fi

        echo "${PROJECT_NAME},${STATUS},${EXIT_CODE},${LOG_FILE}" >> "$SUMMARY_FILE"
    fi
done

echo
echo "-------------------------------------------------------"
echo "TEST SUMMARY"
echo "-------------------------------------------------------"
column -t -s, "$SUMMARY_FILE" || cat "$SUMMARY_FILE"
echo "-------------------------------------------------------"
echo "Full logs (if enabled) are available under: /tmp/test_run_${TIMESTAMP}"
echo "-------------------------------------------------------"

exit $OVERALL_EXIT
