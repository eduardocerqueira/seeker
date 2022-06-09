#date: 2022-06-09T17:14:27Z
#url: https://api.github.com/gists/a84a564441bf222197f020d5e03165ee
#owner: https://api.github.com/users/Aliamondo

#!/usr/bin/env bash
i=1
successes=0
failures=0
totalTests=10
SUCCESS_CHECKMARK=$(printf '\342\234\224\n' | iconv -f UTF-8)
CROSS_MARK=$(printf '\342\235\214\n' | iconv -f UTF-8)
command="yarn test"

if [ -n "$1" ]
then
  command="$@"
else
  command="yarn test"
fi
echo "Running $command $totalTests times"

until [ $i -gt $totalTests ]; do
  echo "Attempt #$i"
  if eval "$command" --silent; then
    ((successes = successes + 1))
    echo "  $SUCCESS_CHECKMARK tests passed"
  else
    ((failures = failures + 1))
    echo "  $CROSS_MARK tests failed"
  fi
  ((i = i + 1))
done

echo ___________________________
echo
echo "Ran $totalTests Tests."
if (($successes > 0))
then
    echo "✅ Succeeded: $successes/$totalTests"
fi
if (($failures > 0))
then
    echo "❌ Failed: $failures/$totalTests"
fi
echo
