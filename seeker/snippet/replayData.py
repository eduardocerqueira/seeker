#date: 2022-03-17T17:12:17Z
#url: https://api.github.com/gists/6464722e2477aaf3cf5eca69308ca0df
#owner: https://api.github.com/users/hafiz703

"""
Read tick-by-tick data for binance's BTC-USDT pair and store it in an sorted dict  
"""
interval = 1
bids = SortedDict()
asks = SortedDict()
cols = ['timestamp_dt', 'bid', 'ask', 'bs', 'as']
df = pd.DataFrame(columns=cols)
line_counter = 0

with gzip.open(fname_book) as f:
  line = f.readline()
  
  prev_snapshot = False
  prev_timestamp = None
  bucket_timestamp = from_ts + (interval * 10**6)
  d = 0
  while line:
    line = f.readline()
    line_counter += 1
    if(line_counter % 1000000 == 0): print(f"Line : {line_counter}")
    if not line: break
    exchange, symbol, timestamp, local_timestamp, is_snapshot, side, price, amount = line.decode('utf-8').strip().split(',')
    price = float(price)
    amount = float(amount)
    local_timestamp = int(local_timestamp)
    
    
    if prev_timestamp != local_timestamp:
      while(local_timestamp > bucket_timestamp and (bids and asks)):
        addRow(df, bucket_timestamp, bids, asks)
        bucket_timestamp += (interval * 10**6)

    prev_timestamp = local_timestamp


    if(is_snapshot and not prev_snapshot):
      print("clearing")
      bids.clear()
      asks.clear()

    prev_snapshot = is_snapshot
    
    if (side == 'bid'):
      if(amount == 0 and price in bids):
        del bids[price]
      
      elif (amount != 0):
        bids[price] = amount
   
    elif (side == 'ask'):
      if(amount == 0 and price in asks):
        del asks[price]

      elif (amount != 0):
        asks[price] = amount  
