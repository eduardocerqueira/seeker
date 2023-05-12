#date: 2023-05-12T16:44:41Z
#url: https://api.github.com/gists/6de3b69601329cc4e0c5bd83f7fce957
#owner: https://api.github.com/users/dbrant

# Dmitry Brant, 2023

import requests
import urllib

def parseDescription(content):
    description = ""
    sStart = content.lower().find("{{short description|")
    if sStart != -1:
        sEnd = content.find("}}", sStart)
        if sEnd != -1:
            description = content[sStart + 20:sEnd]
    return description

def cleanupForCsv(s):
    return s.replace("\"", "”").replace("'", "’").replace(",", " ").replace("\r", " ").replace("\n", "")


continueStr = ""
userEditCounts = {}

file = open("machineedits_en.csv", "w", encoding="utf-8")

while True:

    req = "https://en.wikipedia.org/w/api.php?action=query&format=json&list=recentchanges&formatversion=2&rctag=android%20app%20edit&rcprop=title%7Ctimestamp%7Cids%7Ccomment%7Ctags%7Cuser&rclimit=500&rctype=edit"
    if continueStr != "":
        req += "&rccontinue=" + continueStr
    
    json = requests.get(req).json()

    continueStr = json['continue']['rccontinue']

    if json["query"]["recentchanges"][0]["timestamp"] < "2023-04-13T00:00:00Z":
        break

    for edit in json['query']['recentchanges']:
        if 'comment' not in edit:
            continue

        isSuggestedEdit = "#suggestededit" in edit['comment']
        isMachineSuggested = "#machine" in edit['comment']
        isMachineSuggestedAndModified = "#machine-suggestion-modified" in edit['comment']

        if not isSuggestedEdit and not isMachineSuggested:
            continue

        # Comment out the following to get all suggested edits, not just machine edits.
        if not isMachineSuggested:
            continue

        user = edit['user']
        editcount = 0
        if user not in userEditCounts:
            req = "https://en.wikipedia.org/w/api.php?action=query&format=json&meta=globaluserinfo&formatversion=2&guiuser=" + user + "&guiprop=editcount%7Cgroups%7Crights"
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
        
        title = edit['title']
        timestamp = edit['timestamp']
        lang = "en"
        revid = edit['revid']
        historyUrl = "https://en.wikipedia.org/w/index.php?title=" + urllib.parse.quote(title) + "&action=history"

        
        req = "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=revisions&revids=" + str(revid) + "&formatversion=2&rvprop=ids%7Ctimestamp%7Cflags%7Ccomment%7Cuser%7Ccontent&rvslots=main"
        json2 = requests.get(req).json()
        description = ""
        if 'query' in json2:
            description = parseDescription(json2['query']['pages'][0]['revisions'][0]['slots']['main']['content'])

        req = "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=revisions&pageids=" + str(edit['pageid']) + "&formatversion=2&rvprop=ids%7Ctimestamp%7Cflags%7Ccomment%7Cuser%7Ccontent&rvslots=main"
        json2 = requests.get(req).json()
        currentDescription = ""
        if 'query' in json2:
            currentDescription = parseDescription(json2['query']['pages'][0]['revisions'][0]['slots']['main']['content'])

        rewritten = description != currentDescription

        line = lang + "\t" + timestamp + "\t" + title + "\t" + historyUrl + "\t" + description.replace("\t", " ") + "\t" + user + "\t" + str(editcount) + "\t" + str(reverted) + "\t" + str(rewritten) + "\t" + str(isMachineSuggested) + "\t" + str(isMachineSuggestedAndModified)

        print(line)
        file.write(cleanupForCsv(line) + "\n")
