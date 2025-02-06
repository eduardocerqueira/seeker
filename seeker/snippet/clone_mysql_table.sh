#date: 2025-02-06T17:11:33Z
#url: https://api.github.com/gists/3d9421bca3ded11b2c62b49c94b7dc5e
#owner: https://api.github.com/users/renanfenrich

#!/bin/bash

# Function to show usage
usage() {
    echo "Usage: $0 -h <DB_HOST> -u <DB_USER> -p <DB_PASS> -d <DB_NAME> -s <SRC_TABLE> -t <DEST_TABLE> -b <BATCH_SIZE>"
    exit 1
}

# Default batch size for large table inserts
BATCH_SIZE=10000

# Parse command-line arguments
while getopts ":h:u:p:d:s:t:b:" opt; do
    case $opt in
        h) DB_HOST="$OPTARG" ;;
        u) DB_USER="$OPTARG" ;;
        p) DB_PASS="$OPTARG" ;;
        d) DB_NAME="$OPTARG" ;;
        s) SRC_TABLE="$OPTARG" ;;
        t) DEST_TABLE="$OPTARG" ;;
        b) BATCH_SIZE="$OPTARG" ;;
        *) usage ;;
    esac
done

# Validate required arguments
if [[ -z "$DB_HOST" || -z "$DB_USER" || -z "$DB_PASS" || -z "$DB_NAME" || -z "$SRC_TABLE" || -z "$DEST_TABLE" ]]; then
    usage
fi

# Step 1: Clone the table structure (indexes included)
echo "Cloning table structure..."
mysql -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASS" "$DB_NAME" -e "CREATE TABLE $DEST_TABLE LIKE $SRC_TABLE;"

# Step 2: Optimize for large data inserts (Disable Keys)
echo "Disabling indexes for faster inserts..."
mysql -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASS" "$DB_NAME" -e "ALTER TABLE $DEST_TABLE DISABLE KEYS;"

# Step 3: Copy data in chunks (to prevent locking issues)
echo "Copying data in batches of $BATCH_SIZE rows..."
MAX_ID=$(mysql -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASS" "$DB_NAME" -N -e "SELECT MAX(id) FROM $SRC_TABLE;")
START_ID=1

while [ "$START_ID" -le "$MAX_ID" ]; do
    END_ID=$((START_ID + BATCH_SIZE - 1))
    echo "Copying rows from $START_ID to $END_ID..."
    mysql -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASS" "$DB_NAME" -e "
        INSERT INTO $DEST_TABLE SELECT * FROM $SRC_TABLE WHERE id BETWEEN $START_ID AND $END_ID;
    "
    START_ID=$((END_ID + 1))
done

# Step 4: Re-enable indexes after the data copy
echo "Re-enabling indexes..."
mysql -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASS" "$DB_NAME" -e "ALTER TABLE $DEST_TABLE ENABLE KEYS;"

# Step 5: Set AUTO_INCREMENT (if applicable)
AUTO_INC=$(mysql -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASS" "$DB_NAME" -N -e "SELECT AUTO_INCREMENT FROM information_schema.TABLES WHERE TABLE_SCHEMA='$DB_NAME' AND TABLE_NAME='$SRC_TABLE';")
if [[ ! -z "$AUTO_INC" ]]; then
    echo "Setting AUTO_INCREMENT value..."
    mysql -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASS" "$DB_NAME" -e "ALTER TABLE $DEST_TABLE AUTO_INCREMENT=$AUTO_INC;"
fi

echo "Table $DEST_TABLE successfully cloned from $SRC_TABLE with optimized performance."
