#date: 2021-09-09T17:06:55Z
#url: https://api.github.com/gists/6eda9de6e96d9803c1614995eaa731e6
#owner: https://api.github.com/users/nickfogle

import stripe
stripe.api_key = "sk_test_XXXXXXX"

# Pass customer email to API endpoint
customers = stripe.Customer.list(email='customer@email.com')
print(customers)
print([cus.id for cus in customers.data])

# take cus.id and pass as CUSTOMER_ID in hmac function here https://gist.github.com/nickfogle/82dba67f7e1f730ca2f147a263e35508