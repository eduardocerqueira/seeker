#date: 2023-02-17T16:55:08Z
#url: https://api.github.com/gists/af1472c602736e83a23186c3949ba4f0
#owner: https://api.github.com/users/butlerbt

import pandas as pd

# these are the names of two processes we are currently most interested in:
jvm = "java -Dlog.store=FILE -Xmx128m -XX:+UseSerialGC -XX:TieredStopAtLevel=1 -Droot=/greengrass/v2 -jar /greengrass/v2/alts/current/distro/lib/Greengrass.jar --setup-system-service false"
cloud_pub = "python /var/code/cloud_publisher_v2/start.py"

with open("memory_log.txt", "r") as f:
    lines = f.readlines()

df = pd.DataFrame(columns=["datetime"])

for line in lines:
    if line.startswith("==="):
        # Extract the datetime from the line
        datetime_str = line.strip("=== ").strip(" ===\n")
        datetime_obj = pd.to_datetime(datetime_str)
        row = {"datetime": datetime_obj}
        df = df.append(row, ignore_index=True)
    else:
        # Extract the column name and value from the line
        parts = line.split()
        column_name = " ".join(parts[1:])
        column_value = float(parts[0])

        # Add the column to the dataframe
        df.at[df.index[-1], column_name] = column_value

print(df[["datetime", jvm, cloud_pub]])
df[["datetime", jvm, cloud_pub]].plot()