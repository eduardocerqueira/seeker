#date: 2025-12-31T16:41:31Z
#url: https://api.github.com/gists/501c40bf8e84838f04811ab5ab0da47a
#owner: https://api.github.com/users/cgoldberg

#!/usr/bin/env bash
#
# Selenium - yearly Git activity
#
# author: Corey Goldberg (https://github.com/cgoldberg)
# requires:
#  - git (https://git-scm.com)
#  - git-who (https://github.com/sinclairtarget/git-who)
#  - selenium repo (git clone https://github.com/SeleniumHQ/selenium.git)

set -e


year="2025"
begin_date="01-01-${year}"
end_date="01-01-$((year + 1))"
file_globs=("*.cs" "*.java" "*.js" "*.py" "*.rb" "*.rs" "*.sh")
languages=(".NET" "Java" "JS" "Python" "Ruby" "Rust" "Shell")

echo
echo "GitHub repo: https://github.com/SeleniumHQ/selenium"
echo
echo
echo "All-time stats"
echo "--------------"
first_commit="$(git log --reverse --all --format='format:%at' | head -n 1)"
now="$(date +%s)"
age_secs="$((${now} - ${first_commit}))"
age_days="$((age_secs / 86400))"
echo "Age: $((age_days / 365)) years, $((age_days % 365 / 30)) months"
echo -n "Commits: "
git rev-list --count HEAD \
    | numfmt --grouping
echo -n "Contributors: "
git who -l -csv -n 0 \
    | tail -n +2 \
    | wc -l \
    | numfmt --grouping
echo
echo
echo "Activity in ${year}"
echo "----------------"
echo -n "Commits: "
git rev-list \
    --since "${begin_date}" --until "${end_date}" --count HEAD \
    | numfmt --grouping
echo -n "Contributors: "
git who -l -csv -n 0 -since "${begin_date}" -until "${end_date}" \
    | tail -n +2 \
    | wc -l \
    | numfmt --grouping
echo
echo
echo "Contributors per language in ${year}"
echo "---------------------------------"
for i in "${!file_globs[@]}"; do
    echo -en "${languages[i]}:\t  "
    git who -l -csv -n 0 \
        -since "${begin_date}" -until "${end_date}" "${file_globs[i]}" \
        | tail -n +2 \
        | wc -l \
        | numfmt --grouping
done
echo
echo
echo "Lines modified per language in ${year}"
echo "-----------------------------------"
for i in "${!file_globs[@]}"; do
    echo -en "${languages[i]}:\t "
    git who -l -csv -n 0 \
        -since "${begin_date}" -until "${end_date}" "${file_globs[i]}" \
        | tail -n +2 \
        | awk -F "," \
        '{a+=$3}{r+=$4} END {printf "%\047d (+%\047d/-%\047d)\n", a + r, a, r}'
done
for i in "${!file_globs[@]}"; do
    echo -e "\n\nTop contributors for ${languages[i]} in ${year}:"
    git who -n 5 -nauthor "Selenium CI Bot" \
        -since "${begin_date}" -until "${end_date}" "${file_globs[i]}"
done
