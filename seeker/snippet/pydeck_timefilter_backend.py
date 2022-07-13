#date: 2022-07-13T17:12:19Z
#url: https://api.github.com/gists/ab0832ee9aa3363dd723211867554f7a
#owner: https://api.github.com/users/josephlewisjgl

# time cols for filtering whether a store is open
time_cols = {'Monday': 'mon_hrs', 
    'Tuesday': 'tue_hrs', 
    'Wednesday': 'wed_hrs', 
    'Thursday': 'thu_hrs', 
    'Friday': 'fri_hrs', 
    'Saturday': 'sat_hrs', 
    'Sunday': 'sun_hrs'}

# find the current date to compare
now = dt.now()

# get the current day but at midnight 
midnight_today = date.today()

# day to compare to 
col_to_use = time_cols.get(now.strftime('%A'))

# find whether a DD is open 
def is_open(time_frame):
    
    if time_frame == 'Open 24 Hours':
        return True

    if time_frame == 'Closed':
        return False

    if time_frame is None:
        return False

    # strip the opening times out of that col 
    opens_at = re.search('^[0-9]{1,2}:[0-9]{2} [A-z]{2}', time_frame).group()
    closes_at = re.search('- [0-9]{1,2}:[0-9]{2} [A-z]{2}', time_frame).group()
    
    # strip out the time data 
    opens_at_time = dt.strptime(opens_at, "%I:%M %p").time()
    closes_at_time = dt.strptime(closes_at, "- %I:%M %p").time()

    # combine the hour and day 
    opens_at_dt = dt.combine(midnight_today, opens_at_time)
    closes_at_dt = dt.combine(midnight_today, closes_at_time)

    # check if it's after opening and before closing time 
    if now > opens_at_dt and now < closes_at_dt:
        return True 
    else:
        return False