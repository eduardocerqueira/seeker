#date: 2023-04-11T16:47:19Z
#url: https://api.github.com/gists/ceb0d1787491e86a826548c99a29239c
#owner: https://api.github.com/users/bn-bnelson

"""
Plots the date last opened against date received
"""

import pandas as pd

from sqlalchemy import create_engine
from matplotlib import pyplot as plt


engine = create_engine('sqlite:///~/Library/Mail/V5/MailData/Envelope Index')

with engine.connect() as conn, conn.begin():
    msg = pd.read_sql_table('messages', conn)
    adr = pd.read_sql_table('addresses', conn)
    
for x in ['date_sent', 'date_received', 'date_created', 'date_last_viewed']:    
    msg[x.replace('date_', 'dt_')] = pd.to_datetime(msg[x], unit='s')
    
    
# Plot opened emails in the last 180 days
mask = msg['date_sent'] > 1508958377
msg[mask].plot(x='date_received', y='date_last_viewed', kind='scatter', figsize=(10,10))

plt.title(f'Re-opened emails in the last 180 days')
plt.show()