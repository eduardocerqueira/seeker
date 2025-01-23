#date: 2025-01-23T16:57:41Z
#url: https://api.github.com/gists/a191a83293a9ee67afc444b0ed5d12cb
#owner: https://api.github.com/users/pphatdev

#!/usr/bin/env sh

_() {
    YEAR="2019"
    echo "GitHub Username: "
    read -r USERNAME
    echo "**********"GitHub Access token: "**********"
    read -r ACCESS_TOKEN

    # Validate inputs
    [ -z "$USERNAME" ] && echo "Error: Username cannot be empty" && exit 1
    [ -z "$ACCESS_TOKEN" ] && echo "Error: "**********"

    # Create and enter directory
    [ ! -d "$YEAR" ] && mkdir "$YEAR"
    cd "$YEAR" || { echo "Error: Could not enter directory"; exit 1; }

    # Initialize git and create content
    git init


    # Create multiple commits with different dates
    for month in {01..12}; do
        for day in {01..31}; do
            # Skip invalid dates
            if [ "$month" = "02" ] && [ "$day" -gt "28" ]; then
                continue
            elif [ "$month" = "04" ] && [ "$day" -gt "30" ]; then
                continue
            elif [ "$month" = "06" ] && [ "$day" -gt "30" ]; then
                continue
            elif [ "$month" = "09" ] && [ "$day" -gt "30" ]; then
                continue
            elif [ "$month" = "11" ] && [ "$day" -gt "30" ]; then
                continue
            fi

            echo "**!🤖!**\n${day}-${month}-2024: 📃" > README.md
            git add README.md
            GIT_AUTHOR_DATE="2024-${month}-${day}T18:00:00" \
            GIT_COMMITTER_DATE="2024-${month}-${day}T18:00:00" \
            git commit --allow-empty -m "💻 + 🤖 = PPhatDev"
        done
    done

    # Setup remote and push
    git remote add origin "https: "**********"
    git branch -M main
    git push -u origin main -f || { echo "Error: Failed to push to remote"; exit 1; }

    # Cleanup
    cd ..
    rm -rf "$YEAR"

    echo
    echo "✨ Cool, check your profile now! Your commits should appear in ${YEAR}"
} && _

unset -f _should appear in ${YEAR}"
} && _

unset -f _