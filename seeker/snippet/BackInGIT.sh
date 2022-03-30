#date: 2022-03-30T16:57:03Z
#url: https://api.github.com/gists/0d7411bade07ed6059a01881573f60e1
#owner: https://api.github.com/users/NerdyDeedsLLC

#!/bin/bash
LOADTEXT="
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                    Loading BackInGIT.sh v.0.2...  ✓ DONE                      ║
║  Can't remember which branch you were on 20 minutes ago? Having to jump from  ║
║  one to the next like a coked-up kangaroo wearing moonshoes in a trampoline   ║
║ factory? Just type bk for an interactive, date-stamped guide to your history  ║
║      SYNTAX: bk <optionalPageSize> then follow the on-screen instructions     ║
║                                                                               ║
╚═══════════════《 𝙃𝙚𝙡𝙥/𝙈𝙤𝙧𝙚 𝙄𝙣𝙛𝙤: unavailable at this time 》═════════════════════𝕁
" && printf "$LOADTEXT"

function BackInGIT(){
  [[ "$1" != "" ]] && END=$(awk "BEGIN {print $1  * -1}") || END=-25
  export PAGESIZE=$END
  export PAGE=1
  while :
    do
    clear
    echo "$START-$END"
    printf "OPT BRANCH LAST_ACCESSED\n$(git reflog show --date=short --all -v | grep -v rebase | grep -n checkout | grep -vE "\S{40} to [a-z0-9]{7}$" | sed "s/.*[{]\(.*\)[}].* to \(.*\)/ \2 \1/" | grep . -n)" | column -t -c 3 | HEAD $END | TAIL $PAGESIZE | grep -v "column: line too long"
    
    
    printf "\n ▲       ╔═══╦═══╦═══╗\n"
    printf " ▲       ║ 7 ║ 8 ║ 9 ║  To SELECT/SWITCH to one of the branches\n"
    printf "  ◤      ╠═══╬═══╬═══╣  shown in the list above, type the OPT\n"
    printf "    ◤ ◀︎ ◀︎║ 4 ║ 5 ║ 6 ║  number listed alongside it, and press\n"
    printf "         ╠═══╬═══╬═══╣             ┌───────────┐\n"
    printf "         ║ 1 ║ 2 ║ 3 ║             │   ENTER   │\n"
    printf "         ╚═══╩═══╩═══╝             └───────────┘\n\n"
    printf "     ┌─────┐ GoTo    ┌─────┐ GoTo    ┌─────┐ GoTo    ┌─────┐ \n"
    printf "     │  0  │ First   │ ╺━╸ │ Prev    │ ╺╋╸ │ Next    │  𝐐​​​​  │ Quit\n"
    printf "     └─────┘ Page    └─────┘ Page    └─────┘ Page    └─────┘\n\n"
    
    read -p "[0-9] [-] [+] [Q] Selection? >" PREVBRANCH
    case $PREVBRANCH in
    'q'|'Q') echo "Quitting..."
        break;;
    
    '0')  
            PAGE=1
            END=$PAGESIZE;;

    '+')  
            PAGE=$(expr $PAGE + 1)
            END=$(awk "BEGIN {print $PAGE  * $PAGESIZE}");;
          

    '-')  if [[ $PAGE -gt 1 ]]; then
            PAGE=$(expr $PAGE - 1)
            END=$(awk "BEGIN {print $PAGE  * $PAGESIZE}")
          fi
          ;;

      *) newBranch="$(git reflog show --date=short --all -v | grep -v rebase | grep -n checkout | grep -vE "\S{40} to [a-z0-9]{7}$" | sed "s/.* to \(.*\)/\1/" | sed "s/.* to //g" | head -$PREVBRANCH | tail -1)";
        if [[ "$(git branch -l | grep $newBranch)" != "" ]]; then
            echo -e "\n  Switching to branch $newBranch... (COMMAND: git checkout $newBranch)..."
            git checkout "$newBranch"
            echo -e "DONE.\n\n"
        else
            echo "Cannot load branch $newBranch. It appears to be missing from the local machine."
        fi
        break;;  
    esac
  done
}
alias 'bk'="BackInGIT $@"