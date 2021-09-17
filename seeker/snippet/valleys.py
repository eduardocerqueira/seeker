#date: 2021-09-17T17:07:13Z
#url: https://api.github.com/gists/534579ed0d4d7f22a7ddcc83eca94f2f
#owner: https://api.github.com/users/omokehinde

# An avid hiker keeps meticulous records of their hikes. During the last hike that took exactly  steps, for every step it was noted if it was an uphill, , or a downhill,  step. Hikes always start and end at sea level, and each step up or down represents a  unit change in altitude. We define the following terms:

# A mountain is a sequence of consecutive steps above sea level, starting with a step up from sea level and ending with a step down to sea level.
# A valley is a sequence of consecutive steps below sea level, starting with a step down from sea level and ending with a step up to sea level.
# Given the sequence of up and down steps during a hike, find and print the number of valleys walked through.

# I didn't get the answer
def countingValleys(steps, path):
    valleys, ups, downs = 0, 0, 0
    for i in range(steps):
        if path[i] == 'U':
            ups += 1
            if ups + downs == 0 and downs < 0:
                valleys += 1
                ups, downs = 0, 0
        else:
            downs -= 1
            if ups > 1:
                valleys += 1
                ups, downs = 0, 0
    print(valleys)

countingValleys(8, 'DDUUUUDD')
countingValleys(8, 'UDDDUDUU')
countingValleys(4, 'DDUU')
countingValleys(2, 'DU')