#date: 2025-05-07T16:59:00Z
#url: https://api.github.com/gists/9a5f38f47008d74efaf73b09369cd272
#owner: https://api.github.com/users/acerbetti

#!/bin/bash

# Default configuration
SERVER="localhost"
PORT="8000"
PROMPT='Tell me a short story about a robot and a cat.'
RUNS=5
SHOW_TEXT=false

# Parse command-line options
while [[ $# -gt 0 ]]; do
  case $1 in
    --server)
      SERVER="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --runs)
      RUNS="$2"
      shift 2
      ;;
    --show-text)
      SHOW_TEXT=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

URL="http://$SERVER:$PORT/v1/completions"
TOTAL_TIME=0
TOTAL_TOKENS= "**********"

echo "Prompt: $PROMPT"
echo "Runs: $RUNS"
echo "Server: $SERVER:$PORT"
echo "Show text: $SHOW_TEXT"
echo ""

for i in $(seq 1 $RUNS); do
    echo -n "Run $i... "

    START=$(date +%s.%N)
    RESPONSE=$(curl -s -X POST "$URL" \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "'"$PROMPT"'",
            "max_tokens": "**********"
            "temperature": 0.7
        }')
    END=$(date +%s.%N)

    TEXT=$(echo "$RESPONSE" | jq -r '.completion // .choices[0].text // empty')

    if [[ -z "$TEXT" ]]; then
        echo "❌ No completion returned! Raw response:"
        echo "$RESPONSE"
        TOKENS= "**********"
        TEXT="(no text)"
    else
        TOKENS= "**********"
        DURATION=$(echo "$END - $START" | bc)
        TPS= "**********"
        echo "Time: "**********": $TOKENS, TPS: $TPS"
        if [ "$SHOW_TEXT" = true ]; then
            echo "Response:"
            echo "$TEXT"
        fi
    fi

    echo ""
    TOTAL_TIME=$(echo "$TOTAL_TIME + ($END - $START)" | bc)
    TOTAL_TOKENS= "**********"
done

# Summary
AVG_TIME=$(echo "$TOTAL_TIME / $RUNS" | bc -l)
AVG_TPS= "**********"

echo "--- Benchmark Summary ---"
echo "Average latency: $(printf "%.2f" "$AVG_TIME")s"
echo "Average tokens/sec: "**********"
echo "Total time: $(printf "%.2f" "$TOTAL_TIME")s over $RUNS runs" $(printf "%.2f" "$AVG_TIME")s"
echo "Average tokens/sec: "**********"
echo "Total time: $(printf "%.2f" "$TOTAL_TIME")s over $RUNS runs"over $RUNS runs"