#date: 2025-01-21T17:12:34Z
#url: https://api.github.com/gists/cdb4c914d188c804d2d169c66d240dae
#owner: https://api.github.com/users/tbbooher

import psycopg2
import csv
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv('../.env')

# Database connection parameters
conn_params = {
    "host": os.getenv("DATABASE_HOST"),
    "port": os.getenv("LOCAL_DATABASE_PORT"),
    "database": "amzn",
    "user": os.getenv("DATABASE_USER"),
    "password": "**********"
}

# Path to the CSV file
csv_file_path = "/Users/tim/code/finance_automator/amzn_data/Retail.OrderHistory.1/Retail.OrderHistory.1.csv"

# Helper function to convert values to numeric
def convert_to_numeric(value):
    """Converts a string to a float, handling 'Not Available' and other invalid values."""
    if not value or value.strip().lower() in ["not available", "n/a", "null", ""]:
        return None
    try:
        # Remove commas and dollar signs, then convert to float
        return float(value.replace(",", "").replace("$", "").strip())
    except ValueError:
        return None

# Helper function to convert values to timestamp
def convert_to_timestamp(value):
    """Converts a string to a timestamp, handling invalid or empty values."""
    if not value or value.strip().lower() in ["not available", "n/a", "null", ""]:
        return None
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None

# Main import logic
try:
    conn = psycopg2.connect(**conn_params)
    conn.autocommit = True  # Enable autocommit
    print("Database connection successful")
    cur = conn.cursor()
    
    # Truncate the table to remove existing data
    cur.execute("TRUNCATE TABLE orders")

    # Define which columns are numeric or timestamp
    numeric_columns = [5, 6, 7, 8, 9, 10, 11]  # Indices for numeric fields
    timestamp_columns = [2, 18]  # Indices for timestamp fields

    with open(csv_file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"', escapechar='\\')
        next(reader)  # Skip the header row

        for row_number, row in enumerate(reader, start=1):
            # Convert numeric columns
            row = [
                convert_to_numeric(value) if i in numeric_columns else value
                for i, value in enumerate(row)
            ]
            # Convert timestamp columns
            row = [
                convert_to_timestamp(value) if i in timestamp_columns else value
                for i, value in enumerate(row)
            ]
            # Replace "Not Available" with None in all columns
            row = [None if v in ["Not Available", "n/a", "null", ""] else v for v in row]

            # Adjust row length to match the table schema
            if len(row) > 27:
                row = row[:27]  # Truncate to 27 columns
            elif len(row) < 27:
                row.extend([None] * (27 - len(row)))  # Pad with None

            # Debugging output
            print(f"Row {row_number}: {row}")
            print(f"Row {row_number} Length: {len(row)}")

            # Insert into the database
            try:
                cur.execute("""
                    INSERT INTO orders (
                        website, order_id, order_date, purchase_order_number, currency,
                        unit_price, unit_price_tax, shipping_charge, total_discounts, total_owed,
                        shipment_item_subtotal, shipment_item_subtotal_tax, asin, product_condition,
                        quantity, payment_instrument_type, order_status, shipment_status, ship_date,
                        shipping_option, shipping_address, billing_address, carrier_name_tracking_number,
                        product_name, product_subcategory, product_brand, product_manufacturer
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s
                    )
                """, row)
            except Exception as e:
                print(f"Row {row_number} insertion error: {e}")

except Exception as e:
    print(f"Error connecting to the database: {e}")

finally:
    if 'cur' in locals() and cur:
        cur.close()
    if 'conn' in locals() and conn:
        conn.close()      conn.close()