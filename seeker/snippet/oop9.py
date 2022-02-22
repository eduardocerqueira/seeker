#date: 2022-02-22T17:02:19Z
#url: https://api.github.com/gists/1abc6e843c9b0f1da4221ffaa8e2ebad
#owner: https://api.github.com/users/jimmy-law

    # convert a SportsDB timestamp to localized display string
    def to_display_string(self, timestamp_str):
        # handle some timestamp strings not have the offset due to data quality issue
        if len(timestamp_str)==25:
            dt_orig = dt.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S%z") 
        else:
            dt_orig = dt.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
        ts_orig = dt_orig.timestamp()
        dt_display = dt.datetime.fromtimestamp(ts_orig, self.display_timezone)
        return dt_display.strftime("%d %b %Y %H:%M")