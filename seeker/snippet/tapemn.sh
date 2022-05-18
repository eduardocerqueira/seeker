#date: 2022-05-18T17:20:32Z
#url: https://api.github.com/gists/bf30e41da29c795aa7d6ba501ed2c0d0
#owner: https://api.github.com/users/WCSN

#!/bin/bash
###############################################################
#
# Tape record/read for DAT streamer
#
# wocson (c) | Modify
# Origin: https://l8sae-rexhn1.livejournal.com/15517.html
###############################################################
export TAPE="/dev/nsa0"
DLM="======================================================"

function anykey
{
  read -n 1 -p "Press any key to continue..."
}

function menu()
{
  echo 
  "
   1. Show list of files of current block
   2. Write new data (append tape)
   3. Rewind tape (Set to BOT)
   4. Wind tape (Set to EOD)
   5. Set head to N blocks before
   6. Set head to N blocks after
   7. Extract data from current block
   8. Erase tape

   0. Exit (e/E)
  --------------------------
  "
}

function status()
{
  echo "$DLM"
  mt status                                                                                                                       
  echo "$DLM"
}

while true; do
  clear
  status; menu;
  read -p "Select action: " ans                                                                                                       

  case $ans in
  1)                                                                                                                              
    echo -e "$DLM \nList:"; tar tzv; echo "$DLM"
    echo "Rewinding to the beginning of current block..."
    mt bsf 2; mt fsf                                                                                                            
    echo "Done"
    anykey                                                                                                         
  ;;
  2)
    read -p "Select file or directory: " path                                                                                   
    cd $(dirname $path)
    if [ $? -ne 0 ]; then
      anykey
      continue
    fi
    echo "Positioning to the end of written data..."
    mt eod; tar czv $(basename $path) -C $(dirname $path)
    echo "Done"; anykey                                                                                                         
  ;;
  3)                                                                                                                              
    echo "Rewinding tape..."; mt rewind; echo "Done"; anykey
  ;;
  4)                                                                                                                              
    echo "Winding tape..."; mt eod; echo "Done"; anykey
  ;;
  5)                                                                                                                              
    read -p "Enter number of blocks before to set to: " ans
    mt bsf $(($ans+1)); mt fsf                                                                                                  
    echo "Done"; anykey                                                                                                         
  ;;
  6)                                                                                                                              
    read -p "Enter number of blocks after to set to: " ans
    mt fsf $ans; echo "Done"; anykey                                                                                            
  ;;
  7)                                                                                                                              
    read -p "Enter folder where to extract: " path
    cd $path
    if [ $? -ne 0 ]; then
      anykey                                                                                                                  
      continue
    fi
    read -p "Extract all data from this block? [Y|n]: " ans
    if [ $ans == "n" ]; then
      read -p "Enter file or dir name: " ans                                                                                  
      tar zxpv $ans
    else
      tar zxpv                                                                                                                
    fi
    echo "Done"; anykey                                                                                                         
  ;;
  8)                                                                                                                              
    echo "WARNING! Erasing will destroy ALL data on tape! Continue? [y|n]";
    if [ $ans == "y" ]; then
      echo "Rewinding tape..."; mt rewind;
      echo "Erasing tape. This is quite long operation..."; mt erase; echo
    fi
    anykey                                                                                                                      
  ;;
  0|e|E)                                                                                                                          
    exit 0                                                                                                                      
  ;;
  *)                                                                                                                              
    continue
  ;;
  esac
done
