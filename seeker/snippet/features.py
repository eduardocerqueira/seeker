#date: 2025-10-31T16:45:25Z
#url: https://api.github.com/gists/e652aa178256940d3cf96ba91aefd75c
#owner: https://api.github.com/users/ntkathole

import pandas as pd

entity_df = pd.DataFrame({
    "customer_id": [
        "CUST_000001",
        "CUST_000002",
        "CUST_000003",
        "CUST_000005",
    ],
    "event_timestamp": [
        pd.Timestamp("2025-08-24 10:17:08.512932839"),
        pd.Timestamp("2025-06-18 20:26:06.075684093"),
        pd.Timestamp("2025-07-07 18:18:11.874200063"),
        pd.Timestamp("2025-07-17 02:28:21.294388467"),
    ]
})

features = fs_banking.get_historical_features(
    entity_df=entity_df,
    features=["customer_high_value_txns:high_value_txn_24h"]
).to_df()

print(features)