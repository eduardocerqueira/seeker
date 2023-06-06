#date: 2023-06-06T17:03:22Z
#url: https://api.github.com/gists/d6385d46bb99c6604a0cc23a370ce12c
#owner: https://api.github.com/users/AgentLoneStar007

# Credit AgentLoneStar007 for writing this
# Log-naming system taken from my project PythonBasicUtils - go check that out
# https://github.com/AgentLoneStar007

import os
from datetime import datetime
import platform
import zipfile
import warnings
from pathlib import Path

# Set some vars
operatingSystem = platform.system().lower()
currentUser = os.path.expanduser('~')
if 'windows' in operatingSystem:
    saveLocation = f"{currentUser}/Documents/My Games/Duskers"
elif 'mac' in operatingSystem:
    # Haven't tested macOS, so this could be wrong
    saveLocation = f"{currentUser}/Library/Application Support/unity.Misfits Attic/Duskers"
else:
    saveLocation = f"{currentUser}/.config/unity3d/Misfits Attic/Duskers"


# Enter the directory you wish to store backups in here - use {currentUser} in the string to get your user directory.
# Example: "{currentUser}/Documents/Duskers Backups/"
backupDir = f"null"


def checkBackupDir(inpt):
    # If the user hasn't bothered to set the backup directory string...
    if inpt == 'null':
        # ...and if there is no persistent text file containing a path...
        if os.path.exists('backupDirectory.txt'):
            with open('backupDirectory.txt', 'r') as file:
                backupDir = file.read().rstrip()
                file.close()
                if os.path.exists(backupDir):
                    return backupDir
                print(f'Backup directory "{backupDir}" does not exist.')
                a = input('Would you like to attempt to create it?')
                if a.lower() == 'y' or a.lower() == '':
                    try:
                        path = Path(backupDir)
                        path.mkdir(parents=True)
                        return backupDir
                    except Exception as error:
                        print(f'Process failed with following error:\n{error}')
                        input('Press Enter to exit...')
                        quit()
                else:
                    input('Cannot continue without backup directory. Press Enter to exit...')
                    quit()

        # ...give a bit of dialogue on creating a directory.
        print('You haven\'t set a backup directory yet!')
        print('Enter the backup directory you would like to use. Type the "~" char for your current user directory.')
        x = input('> ')
        # Replace the "~" in the string(if present) with the user's home directory
        if '~' in x:
            x = x.replace('~', currentUser)
        # If the path given doesn't exist, attempt creation
        if not os.path.exists(x):
            print(f'Backup directory "{x}" does not exist.')
            a = input('Would you like to attempt to create it?')
            if a.lower() == 'y' or a.lower() == '':
                # Try to create the directory given
                try:
                    path = Path(x)
                    path.mkdir(parents=True)
                # If failed, exit
                except Exception as error:
                    print(f'Process failed with following error:\n{error}')
                    input('Press Enter to exit...')
                    quit()
            # If the user denies path creation, exit
            else:
                input('Cannot continue without backup directory. Press Enter to exit...')
                quit()
        print(f'New Backup Directory: {x}')
        # Ask if the user wishes to save the path to a file to avoid this dialogue
        y = input('Would you like to save this directory to a persistent file? <Y/n>')
        if y.lower() == 'y' or y.lower() == '':
            with open('backupDirectory.txt', 'a') as z:
                z.write(f'{x}')
                z.close()
            print('Saved.')
            return x
        else:
            # If they deny, continue
            print('Continuing without saving...')
            return
    # Make sure the path given exists. If it does continue
    if os.path.exists(inpt):
        return inpt

    # If the path doesn't exist, follow the regular dialogue to attempt its creation
    print(f'Backup directory "{inpt}" does not exist.')
    a = input('Would you like to attempt to create it?')
    if a.lower() == 'y' or a.lower() == '':
        try:
            path = Path(inpt)
            path.mkdir(parents=True)
            return inpt
        except Exception as error:
            print(f'Process failed with following error:\n{error}')
            input('Press Enter to exit...')
            quit()
    else:
        input('Cannot continue without backup directory. Press Enter to exit...')
        quit()


# A simple file sorter for the given list of existing backups in the next function
def fileSort(inpt):
    if '-' in inpt[25:]:
        return int(inpt[inpt.index("-", 25) + 1:-4])
    else:
        return 0


def getFileName(inpt):
    # Set some vars
    now = datetime.now()
    date = now.strftime('%m-%d-%Y')
    if inpt.endswith('/') or inpt.endswith('\\'):
        inpt = inpt[:-1]

    # Create a list of every backup in the backup directory.
    existingBackups = []
    for x in os.listdir(inpt):
        if x.endswith('.zip'):
            try:
                # Only add it if the first date given(month) is between 1 and 12. This will work even if the file
                # doesn't have the correct naming scheme
                if 1 <= int(x[8:10]) <= 12:
                    existingBackups.append(x)
            except:
                continue

    # If there are any backups in the backup folder...
    if len(existingBackups) > 0:
        # Create another array, and append all backups in the existingBackups array in it that have the same date
        backupsWithCurrentDate = []
        for x in existingBackups:
            if date in x:
                backupsWithCurrentDate.append(x)
        # If there is only one backup in the list, return the name with a backup number of 1
        if len(backupsWithCurrentDate) == 1:
            del existingBackups
            del backupsWithCurrentDate
            # Return the backup name and its location
            backupName = f'Duskers {date} Backup-1.zip'
            return backupName, f'{inpt}/{backupName}'
        # Otherwise, return a backup name with a number one higher than the last log
        elif len(backupsWithCurrentDate) > 1:
            # Sort the array so that backups in the backupsWithCurrentDate array are ordered by their number at the end
            backupsWithCurrentDate.sort(key=fileSort)
            # Create a var of the name of the last created backup
            currentDateBackup = backupsWithCurrentDate[-1]
            # Set the backup number to be appended at the end of the backup name to be plus one from the last backup
            logNumber = f'{int(currentDateBackup[(currentDateBackup.index("-", 25) + 1):-4]) + 1}'
            # Delete the arrays to save memory - pretty pointless because this program should run in under a second
            del existingBackups
            del backupsWithCurrentDate
            backupName = f'Duskers {date} Backup-{logNumber}.zip'
            # Return the backup name and its location
            return backupName, f'{inpt}/{backupName}'
    # See above comment for array deletion
    del existingBackups
    backupName = f'Duskers {date} Backup.zip'
    # Return the backup name and its location
    return backupName, f'{inpt}/{backupName}'


def zipFiles(directory_path, zip_path):
    # For some reason, this function returns a ton of warnings, so I disabled errors for it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Create the zipfile object
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk the entire directory tree
            for root, _, files in os.walk(directory_path):
                # Zip all files...
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, directory_path))

                # ...and directories(even if they're empty).
                for dir_name, _, _ in os.walk(directory_path):
                    dir_path = os.path.join(root, dir_name)
                    zipf.write(dir_path, os.path.relpath(dir_path, directory_path))


# Finally, the primary function
def main(inpt, saveLocation):
    # Create the needed vars by running the functions above
    backupDir = checkBackupDir(inpt)
    backupName, backupLocation = getFileName(backupDir)
    # Zip the files
    zipFiles(saveLocation, backupLocation)
    # Give a completion dialogue for confirmation, and a manual exit, so you can see the message
    print(f'Zipped contents of "{saveLocation}" to "{backupName}" at "{backupLocation}".')
    input('Press Enter to exit...')
    # Pointless return statement
    return


main(backupDir, saveLocation)
