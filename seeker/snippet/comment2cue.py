#date: 2025-01-15T17:06:29Z
#url: https://api.github.com/gists/bfea9c28fa85ed86955d3e7defd8830d
#owner: https://api.github.com/users/g30r93g

import re

if __name__ == '__main__':
    # setup cuesheet entries
    cuesheetEntries = []

    # obtain comment
    chapterComment = []
    # commentFilename = input("Comment filename (txt) > ")
    # with open(f"./{commentFilename}", 'r') as file:
    with open("./comment.txt", 'r') as file:
        chapterComment = file.readlines()

    # extract each line and obtain timestamp, title and performers
    for idx, line in enumerate(chapterComment):
        # print(line)
        pattern = r"((\d{2}:\d{2})|(\d{1}:\d{2}:\d{2}))\s+(.+)"

        match = re.match(pattern, line)

        timestamp = match.group(1).split(":")
        timestampHours = int(timestamp[0]) if len(timestamp) >= 3 else 0
        timestampMins = int(timestamp[1]) if len(timestamp) >= 3 else int(timestamp[0])
        timestampSecs = int(timestamp[2]) if len(timestamp) >= 3 else int(timestamp[1])

        trackInfo = match.group(4)

        try:
            splitIdx = trackInfo.index("-")

            title = trackInfo[splitIdx+1:].strip()
            performers = trackInfo[:splitIdx].strip()
        except ValueError:
            # If no '-' or '–' found, assume no performers
            title = trackInfo.strip()
            performers = ""

        # add new cuesheet entry
        newCueEntry = f"    TRACK {(idx + 1):02d} AUDIO\n" \
            f"      TITLE \"{title} (Mixed)\"\n" \
            f"      PERFORMER \"{performers}\"\n" \
            f"      INDEX 01 {((timestampHours * 60) + timestampMins):02d}:{timestampSecs:02d}:00\n"

        cuesheetEntries += newCueEntry

    # obtain file meta
    print("Final Details:")
    overallPerformer = input(" Overall Performer > ")
    overallTitle = input(" Overall Title > ")
    audioFilename = input(" Audio Filename > ")

    # create cuesheet
    cuesheetOutput = f"PERFORMER \"{overallPerformer}\"\n" \
        f"TITLE \"{overallTitle}\"\n" \
        f"FILE \"{audioFilename}\" M4A\n"
    
    for entry in cuesheetEntries:
        cuesheetOutput += entry

    print(cuesheetOutput)

    # write cuesheet to file
    with open(f"./{overallTitle.lower().replace(" ", "_").replace("\t", "_")}___{overallPerformer.lower().replace(" ", "_").replace("\t", "_")}.cue", "w") as cueFile:
        cueFile.write(cuesheetOutput)
