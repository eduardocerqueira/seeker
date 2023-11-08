#date: 2023-11-08T16:58:53Z
#url: https://api.github.com/gists/681a5f8c92beea68e13876b2d5f6412f
#owner: https://api.github.com/users/KeithDinh

#! /usr/bin/env bash
# Author: last modified by Kiet Dinh, credits to all people from https://gist.github.com/TheNetJedi/0847ba0ec1da9879b4fa1d8f3276b004
# You need to have aws-cli installed and configured
# You need to have CURL installed

# -------------------------------------------------------

# Explanation:
# mkdir -p: not throwing error if folder exists
# aws lambda list-functions: output metadata of each function (name, arn, description, arn, ...)
	# Ex: 
	# {
	#		"FunctionName" : "Testing",
	#		"ARN" : "Blabla",
	# }
# grep FunctionName: return lines containing string "FunctionName"
	# Ex: "FunctionName" : "Testing"
# cut -d '"' -f4: parse lines with double-quotation delimeter, return the 4th string
	# Ex: "FunctionName" : "Testing" => Testing 
		# f1: 
		# f2: FunctionName
		# f3:  : 
		# f4: Testing
		# f5: 
		
# From what I understand, the output from the "cut -d '"' -f4" will serve as name in the while condition, and the $name inside while loop is each item of the list
# read -r name: read argument as name? The -r means "Disable backslashes to escape characters."
	# aws lambda get-function --function-name: Returns function info with a link to download the deployment package that's valid for 10 minutes
	# tail -n 3: return last n number of line from data
	# egrep: equivalent to grep -E, allows the use of extended regex
		# -o: Outputs only the matched string and not the entire line
		
	# sed: stream editor for filtering and transforming text. 
		# egrep -o 'https?://[^ ]+' returns string with a double quotation at the end such as https://blablabla?dadsf=asdfsdf" 
		# sed 's/"//' strips that last double quotation
	
	# tee: write output to file
# -------------------------------------------------------
	
mkdir -p code
aws --profile yourProfile --region us-east-1 lambda list-functions | \
grep FunctionName | \
cut -d '"' -f4 | \
while read -r name; do
	aws --profile yourProfile lambda get-function --function-name $name | tee ./code/$name.json | egrep -o 'https?://[^ ]+' | sed 's/"//' | xargs curl --output ./code/$name.zip
  # alternative: aws --profile yourProfile lambda get-function --function-name $name | tail -n 3 | egrep -o 'https?://[^ ]+' | sed 's/"//' | xargs curl --output ./code/$name.zip
	# alternative: replace curl with wget
done

