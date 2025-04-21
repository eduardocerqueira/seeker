#date: 2025-04-21T16:39:13Z
#url: https://api.github.com/gists/87a07d5cf409eb7422a739edc6afd627
#owner: https://api.github.com/users/calilisantos

: <<'START_COMMENT'
CLIENT URL (curl) SHELL SCRIPT USE CASES
1. GIVING SCRIPT EXECUTE PERMISSION:
chmod +x client_url_shell.sh
2. RUNNING THE SCRIPT:
./client_url_shell.sh
3. RUNNING THE SCRIPT WITH DEBUGGING:
bash -xv client_url_shell.sh
START_COMMENT

###################

echo "[INFO] 1. WEB PAGE HEALTH CHECK (curl scraping)"
NAME_TO_TEST="Calili Santos"
PROPERTY_TO_TEST="itemprop=\"name\"" # Backslash (\) used to escape the double quotes inside the string

: <<'CONTENT_EXPECTED_REGEX_COMMENT' 
CONTENT_EXPECTED_REGEX will search for the name in the HTML using grep
\s* matches any amount of whitespace (spaces, tabs, newlines)
CONTENT_EXPECTED_REGEX_COMMENT
CONTENT_EXPECTED_REGEX="$PROPERTY_TO_TEST>\s*$NAME_TO_TEST\s*"

URL="https://github.com/calilisantos"
HTML_FLAT=$(curl -s "$URL" | tr -d '\n') # tr -d '\n' removes newlines from the HTML content

if ! echo "$HTML_FLAT" | grep -Pq "$CONTENT_EXPECTED_REGEX"; then
    # grep -Pq: -P enables Perl-compatible regex, -q suppresses output
    echo "[ERROR] $NAME_TO_TEST not found in $URL"
    exit 1
fi
echo "[INFO] Name $NAME_TO_TEST found in $URL"