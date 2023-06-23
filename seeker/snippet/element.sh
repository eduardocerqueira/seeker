#date: 2023-06-23T16:44:49Z
#url: https://api.github.com/gists/3f39d2816c7f82c7484c11124911b234
#owner: https://api.github.com/users/dschlosser

#!/bin/bash

PSQL="psql --username=freecodecamp --dbname=periodic_table -t --no-align -c"

# Abort with message if no args where given
if [[ -z $1 ]]; then
	echo -e "Please provide an element as an argument."
	exit
fi
# If arg might be a number, search by atomic number
if [[ $1 < 'a' ]]; then
	ATOMIC_NUMBER="$($PSQL "SELECT atomic_number FROM elements WHERE atomic_number = CAST('$1' AS INT);")"
else
	# Search by element symbol
	ATOMIC_NUMBER="$($PSQL "SELECT atomic_number FROM elements WHERE symbol = '$1';")"
	# Search by element name
	if [[ -z $ATOMIC_NUMBER ]]; then
		ATOMIC_NUMBER="$($PSQL "SELECT atomic_number FROM elements WHERE name = '$1';")"
	fi
fi
# Element not found
if [[ -z $ATOMIC_NUMBER ]]; then
  echo "I could not find that element in the database."
  exit
fi

# Get element record
ELEMENT_NAME="$($PSQL "SELECT name FROM elements WHERE atomic_number=$ATOMIC_NUMBER;")"
ELEMENT_SYMBOL="$($PSQL "SELECT symbol FROM elements WHERE atomic_number=$ATOMIC_NUMBER;")"
ELEMENT_TYPE_ID="$($PSQL "SELECT type_id FROM properties WHERE atomic_number=$ATOMIC_NUMBER;")"
ELEMENT_TYPE="$($PSQL "SELECT type FROM types WHERE type_id=$ELEMENT_TYPE_ID;")"
ATOMIC_MASS="$($PSQL "SELECT atomic_mass FROM properties WHERE atomic_number=$ATOMIC_NUMBER;")"
MELTING_POINT="$($PSQL "SELECT melting_point_celsius FROM properties WHERE atomic_number=$ATOMIC_NUMBER;")"
BOILING_POINT="$($PSQL "SELECT boiling_point_celsius FROM properties WHERE atomic_number=$ATOMIC_NUMBER;")"

# RESPONSE="$($PSQL "SELECT * FROM elements FULL JOIN properties ON elements.atomic_number = properties.atomic_number FULL JOIN types ON properties.type_id = types.type_id WHERE elements.atomic_number=$ATOMIC_NUMBER;")"
# echo "$RESPONSE" | IFS='|' read NUMBER ELEMENT_SYMBBOL ELEMENT_NAME NUMBER ATOMIC_MASS MELTING_POINT BOLING_POINT TYPE_ID TYPE_ID TYPE

echo -e "The element with atomic number $ATOMIC_NUMBER is $ELEMENT_NAME ($ELEMENT_SYMBOL). It's a $ELEMENT_TYPE, with a mass of $ATOMIC_MASS amu. $ELEMENT_NAME has a melting point of $MELTING_POINT celsius and a boiling point of $BOILING_POINT celsius."
