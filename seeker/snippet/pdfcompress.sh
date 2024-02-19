#date: 2024-02-19T16:54:38Z
#url: https://api.github.com/gists/30c5c2decd0408e61c43160780d0a60c
#owner: https://api.github.com/users/passos

inputFile=$1
outputFile=$(echo $inputFile | sed 's|\.pdf$|\.c\.pdf|ig')

echo " input is $inputFile, \noutput is $outputFile"

while true; do
    # Prompt the user
    read -p "Do you want to continue? (Y/N): " answer

    # Convert the input to lowercase
    answer=$(echo "$answer" | tr '[:upper:]' '[:lower:]')

    # Check the input
    if [[ $answer == "y" ]]; then
        break
    elif [[ $answer == "n" ]]; then
        exit
    fi
done

gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook -dNOPAUSE -dBATCH -sOutputFile="$outputFile" "$inputFile"