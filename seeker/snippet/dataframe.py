#date: 2022-05-25T17:08:27Z
#url: https://api.github.com/gists/cde573e67cf4775523d31c51d29dcd13
#owner: https://api.github.com/users/srang992

# selecting the necessary columns from the dataframe
query = """SELECT complains, charge_amount, seconds_of_use, 
frequency_of_use, frequency_of_sms, age_group, customer_value, churn 
FROM customer_churn"""

tel_data = pd.read_sql(query, conn)