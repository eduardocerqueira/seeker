#date: 2024-11-18T17:01:23Z
#url: https://api.github.com/gists/2c6f00ff5a9f8373ac9df5683329a7d0
#owner: https://api.github.com/users/jac18281828

import os
import subprocess

start_date='2024:10:16'
fixed_date='2024:11:16'

def main():
    # Get all files in the current directory
    files = os.listdir('.')
    for filename in files:
        # Skip directories
        if os.path.isdir(filename):
            continue

        # Run exiftool to get the dates
        cmd = ['exiftool', '-createdate', '-modifydate', '-datetimeoriginal', filename]
        try:
            output = subprocess.check_output(cmd, universal_newlines=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing file {filename}: {e}")
            continue

        # Initialize date variables
        createdate = None
        modifydate = None
        datetimeoriginal = None

        # Parse the output to get the date strings
        for line in output.strip().split('\n'):
            if 'Create Date' in line:
                createdate = line.split(':', 1)[1].strip()
            elif 'Modify Date' in line:
                modifydate = line.split(':', 1)[1].strip()
            elif 'Date/Time Original' in line:
                datetimeoriginal = line.split(':', 1)[1].strip()

        # Replace start_date with fixed_date in the date strings
        if createdate and start_date in createdate:
            new_createdate = createdate.replace(start_date, fixed_date)
        else:
            new_createdate = None

        if modifydate and start_date in modifydate:
            new_modifydate = modifydate.replace(start_date, fixed_date)
        else:
            new_modifydate = None

        if datetimeoriginal and start_date in datetimeoriginal:
            new_datetimeoriginal = datetimeoriginal.replace(start_date, fixed_date)
        else:
            new_datetimeoriginal = None

        # Prepare the exiftool command to update the dates
        cmd = ['exiftool']
        if new_createdate:
            cmd.append(f'-createdate="{new_createdate}"')
        if new_modifydate:
            cmd.append(f'-modifydate="{new_modifydate}"')
        if new_datetimeoriginal:
            cmd.append(f'-datetimeoriginal="{new_datetimeoriginal}"')
        cmd.append(filename)

        # Update the dates if there are any changes
        if len(cmd) > 2:
            try:
                subprocess.check_output(cmd, universal_newlines=True)
                print(f"Updated dates for file {filename}")
            except subprocess.CalledProcessError as e:
                print(f"Error updating file {filename}: {e}")
        else:
            print(f"No dates to update for file {filename}")

if __name__ == "__main__":
    main()
