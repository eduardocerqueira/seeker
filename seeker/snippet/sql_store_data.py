#date: 2022-12-23T17:10:36Z
#url: https://api.github.com/gists/bc0e08b6ef6ee8adba504094dcf988a7
#owner: https://api.github.com/users/DanEdens

import sqlite3

def store_data(device, version, results_url, ticket_key, timestamp, test_name, test_config):
    conn = sqlite3.connect('tests.db')
    c = conn.cursor()
    
    c.execute('CREATE TABLE IF NOT EXISTS tests (device text, version text, results_url text, ticket_key text, timestamp text, test_name text, test_config text)')
    c.execute('INSERT INTO tests (device, version, results_url, ticket_key, timestamp, test_name, test_config) VALUES (?, ?, ?, ?, ?, ?, ?)', (device, version, results_url, ticket_key, timestamp, test_name, test_config))
    
    conn.commit()
    conn.close()
# This function will create a SQLite database called tests.db if it doesn't already exist, and then create a table called tests if it doesn't exist. The function will then insert the provided data into this table.

# You can call this function by passing in the relevant data as arguments, like so:

# Copy code
# store_data('device1', '1.0', 'http://results.com', 'ABC123', '2022-12-09 12:00:00', 'test1', 'config1')