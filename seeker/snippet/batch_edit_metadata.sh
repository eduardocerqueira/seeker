#date: 2023-09-06T17:07:55Z
#url: https://api.github.com/gists/be88576322aeba9d9a1ab845f2aa33b5
#owner: https://api.github.com/users/kokkytos

#! /usr/bin/env bash 
while IFS="," read -r Filename  Title Author  Subject
do
  echo "Filename:$Filename"
  echo "Title: $Title"
  echo "Author: $Author"
  echo "Subject: $Subject"
  echo ""
  exiftool -Title="$Title" -Author="$Author" -Subject="$Subject" "$Filename"

done < <(tail -n +2 titles.csv)


# CSV file (titles.csv) contents: 

#Filename,Title,Author,Subject
#Adelphoi Karamazob - A - Phiontor Ntostogephsku.pdf,Αδελφοί Καραμαζόφ (Α),Fyodor Dostoevsky,Λογοτεχνία