#date: 2025-06-17T16:48:55Z
#url: https://api.github.com/gists/4503c20f4bd8cecb5ff92d1ac435eb80
#owner: https://api.github.com/users/philipjohn

#!/bin/bash

# iterate-by-date.sh - A template script for looping through days/months/years and running a command that takes
# before and after arguments.
# 
# Usage: ./iterate-by-date.sh [iterate_by] [start_date] [end_date]
# 
# Options:
# - iterate_by: The granularity of the operation - day, month or year (default: day).
# - start_date: The start date for the operation in YYYY-MM-DD format (default: 30 days ago).
# - end_date: The end date for the operation in YYYY-MM-DD format (default: today).

# The first argument determines whether we loop through the dates by day, month or year.
ITERATE_BY=$1
# If iterate_by is not provided, default to day.
if [[ -z "$ITERATE_BY" ]]; then
	ITERATE_BY="day"
else
	case "$ITERATE_BY" in
		day|month|year)
			;;
		*)
			echo "Invalid iterate_by. Valid options are: day, month, year"
			exit 1
			;;
	esac
fi

# Function to get the next before/after dates based on the iterate_by option.
next_date() {
	local THE_DATE=$1
	local THE_TYPE=$2

	if [ "$THE_TYPE" == "after" ]; then

		NEW_DATE=$(gdate -d "$THE_DATE -1 second" +%Y-%m-%d)
		if [ "$ITERATE_BY" == "year" ]; then
			NEW_DATE=$(gdate -d "$THE_DATE -1 year" +%Y-%m-%d)
		elif [ "$ITERATE_BY" == "month" ]; then
			NEW_DATE=$(gdate -d "$THE_DATE" +%Y-%m-01)
			NEW_DATE=$(gdate -d "$NEW_DATE -1 second" +%Y-%m-%d)
		fi
		
	elif [ "$THE_TYPE" == "before" ]; then
		NEW_DATE=$(gdate -d "$THE_DATE -1 second" +%Y-%m-%d)
		if [ "$ITERATE_BY" == "year" ]; then
			NEW_DATE=$(gdate -d "$NEW_DATE +1 second" +%Y-01-01)
		elif [ "$ITERATE_BY" == "month" ]; then
			NEW_DATE=$(gdate -d "$NEW_DATE +1 second" +%Y-%m-01)
		fi
	fi

	echo $NEW_DATE
}

# The second and third arguments, when provided, are the start date and end date.
START_DATE=$2
END_DATE=$3

# If we have no given start and end date, default to -30 days ago and today.
if [[ -z "$START_DATE" ]]; then
	START_DATE=$(gdate -d "-30 days" +%Y-%m-%d)
fi
if [[ -z "$END_DATE" ]]; then
	END_DATE=$(gdate +%Y-%m-%d)
fi

# Validate the date format.
if ! gdate -d "$START_DATE" &>/dev/null; then
	echo "Invalid start date format. Please use YYYY-MM-DD."
	exit 1
fi
if ! gdate -d "$END_DATE" &>/dev/null; then
	echo "Invalid end date format. Please use YYYY-MM-DD."
	exit 1
fi

# Construct the initial before date argument.
BEFORE=$(gdate -d "$END_DATE +1 day" +%Y-%m-%d)
if [[ "$ITERATE_BY" == "year" ]]; then
	BEFORE=$(gdate -d "$END_DATE +1 year" +%Y-01-01)
elif [[ "$ITERATE_BY" == "month" ]]; then
	BEFORE=$(gdate -d "$END_DATE" +%Y-%m-01)
	BEFORE=$(gdate -d "$BEFORE +1 month" +%Y-%m-%d)
fi

# Construct the initial after date argument.
AFTER=$(gdate -d "$END_DATE -1 second" +%Y-%m-%d)
if [[ "$ITERATE_BY" == "year" ]]; then
	AFTER=$(gdate -d "$END_DATE -1 year" +%Y-12-31)
elif [[ "$ITERATE_BY" == "month" ]]; then
	AFTER=$(gdate -d "$END_DATE" +%Y-%m-01)
	AFTER=$(gdate -d "$AFTER -1 second" +%Y-%m-%d)
fi

# Run through each date until $AFTER is before $START_DATE.
while [[ "$(gdate -d "$AFTER" +%s)" -ge "$(gdate -d "$START_DATE -1 $ITERATE_BY" +%s)" ]];
do
	# Generate the name of the output file.
	OUTPUT_FILE_DATE_FORMAT="+%Y-%m-%d"
	if [[ "$ITERATE_BY" == "year" ]]; then
		OUTPUT_FILE_DATE_FORMAT="+%Y"
	elif [[ "$ITERATE_BY" == "month" ]]; then
		OUTPUT_FILE_DATE_FORMAT="+%Y-%m"
	fi
	OUTPUT_FILE_DATE=$(next_date "$BEFORE" "before")
	OUTPUT_FILE_DATE=$(gdate -d "$OUTPUT_FILE_DATE" "$OUTPUT_FILE_DATE_FORMAT")
	OUTPUT_FILE="logs/${APP}-${ENV}-${OPERATION}-${OUTPUT_FILE_DATE}.$FORMAT"

	START_TIME=$(gdate +%s)

	echo "Running command for date range: $AFTER to $BEFORE" | tee -a $LOG_NAME
	echo "  Command is: "**********"
	echo "  Output will be saved to: $OUTPUT_FILE" | tee -a $LOG_NAME
	
	# Replace the following line with your command.
	echo "--after=$AFTER --before=$BEFORE" | tee $OUTPUT_FILE

	END_TIME=$(gdate +%s)
	TIME_DIFF=$((END_TIME - START_TIME))
	echo "  Command took $TIME_DIFF seconds." | tee -a $LOG_NAME

	BEFORE=$(next_date "$BEFORE" "before")
	AFTER=$(next_date "$AFTER" "after")
done
s." | tee -a $LOG_NAME

	BEFORE=$(next_date "$BEFORE" "before")
	AFTER=$(next_date "$AFTER" "after")
done
