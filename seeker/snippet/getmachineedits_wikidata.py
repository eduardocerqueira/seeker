#date: 2023-05-12T16:45:48Z
#url: https://api.github.com/gists/33d72c7610fc42d357ccbab57d31ef2d
#owner: https://api.github.com/users/dbrant

# Dmitry Brant, 2023

import requests

def cleanupForCsv(s):
    return s.replace("\"", "”").replace("'", "’").replace(",", " ").replace("\r", " ").replace("\n", "")

apiBaseUrl = "https://www.wikidata.org/w/api.php?format=json&formatversion=2&"
continueStr = ""
userEditCounts = {}

file = open("machineedits.csv", "w", encoding="utf-8")

while True:

    req = apiBaseUrl + "action=query&list=recentchanges&rctag=android app edit&rcprop=title|timestamp|ids|comment|tags|user&rclimit=500&rctype=edit"
    if continueStr != "":
        req += "&rccontinue=" + continueStr
    
    json = requests.get(req).json()

    continueStr = json['continue']['rccontinue']

    if json["query"]["recentchanges"][0]["timestamp"] < "2023-04-13T00:00:00Z":
        break

    for edit in json['query']['recentchanges']:
        isMachineSuggested = "#machine" in edit['comment']
        isMachineSuggestedAndModified = "#machine-suggestion-modified" in edit['comment']

        # Comment out the following line to get all edits, not just machine edits.
        if not isMachineSuggested:
            continue

        user = edit['user']
        editcount = 0
        if user not in userEditCounts:
            req = apiBaseUrl + "action=query&meta=globaluserinfo&guiuser=" + user + "&guiprop=editcount|groups|rights"
            json2 = requests.get(req).json()
            if "query" in json2 and "editcount" in json2['query']['globaluserinfo']:
                editcount = json2['query']['globaluserinfo']['editcount']
            userEditCounts[user] = editcount
        else:
            editcount = userEditCounts[user]

        reverted = False
        for tag in edit['tags']:
            if "revert" in tag or "undo" in tag or "rollback" in tag:
                reverted = True
        
        timestamp = edit['timestamp']
        lang = edit["comment"].split(" ")[1].split("|")[1]
        description = edit["comment"].split("*/ ")[1].split(", #")[0]
        title = edit['title']
        historyUrl = "https://www.wikidata.org/w/index.php?title=" + title + "&action=history"

        req = apiBaseUrl + "action=wbgetentities&ids=" + title
        json2 = requests.get(req).json()
        currentDescription = ""
        if 'descriptions' in json2['entities'][title] and lang in json2['entities'][title]['descriptions']:
            currentDescription = json2['entities'][title]['descriptions'][lang]['value']
        
        rewritten = description != currentDescription

        label = title
        if 'labels' in json2['entities'][title] and lang in json2['entities'][title]['labels']:
            label = json2['entities'][title]['labels'][lang]['value']
        
        line = lang + "\t" + timestamp + "\t" + label + "\t" + historyUrl + "\t" + description.replace("\t", " ") + "\t" + user + "\t" + str(editcount) + "\t" + str(reverted) + "\t" + str(rewritten) + "\t" + str(isMachineSuggested) + "\t" + str(isMachineSuggestedAndModified)
        print(line)
        file.write(cleanupForCsv(line) + "\n")
