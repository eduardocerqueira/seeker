#date: 2026-01-12T17:14:06Z
#url: https://api.github.com/gists/14fae59c71921710a3e055d74f30c8af
#owner: https://api.github.com/users/prateek

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
	echo "Usage: $0 <iterations>"
	exit 1
fi

iterations="$1"

if ! [[ "$iterations" =~ ^[0-9]+$ ]] || [[ "$iterations" -lt 1 ]]; then
	echo "Iterations must be a positive integer"
	exit 1
fi

if ! command -v codex >/dev/null 2>&1; then
	echo "Error: codex CLI not found on PATH"
	exit 1
fi

if [[ ! -f "PRD.md" ]] || [[ ! -f "progress.txt" ]]; then
	echo "Error: expected PRD.md and progress.txt in the repo root."
	echo "Create them first (see README.md), then re-run."
	exit 1
fi

PROMISE_FILE="I_PROMISE_ALL_TASKS_IN_THE_PRD_ARE_DONE_I_AM_NOT_LYING_I_SWEAR"

mkdir -p .logs
rm -f "$PROMISE_FILE"

for ((i = 1; i <= iterations; i++)); do
	result="$(
		codex --dangerously-bypass-approvals-and-sandbox exec <<'EOF' 2>&1
1. Find the highest-priority task based on PRD.md and progress.txt, and implement it.
2. Run your tests and type checks.
3. Update the PRD with what was done.
4. Append your progress to progress.txt.
5. Commit your changes.
ONLY WORK ON A SINGLE TASK.

If the PRD is complete, and there are NO tasks left, then and only then touch a file named I_PROMISE_ALL_TASKS_IN_THE_PRD_ARE_DONE_I_AM_NOT_LYING_I_SWEAR. Otherwise respond with a brief summary of changes/progress.
EOF
	)"
	printf '%s\n' "$result" | tee -a ".logs/iterations.log"

	if [[ -f "$PROMISE_FILE" ]]; then
		echo "PRD complete after $i iterations."
		exit 0
	fi
done

echo "PRD not complete after $iterations iterations."
exit 1

