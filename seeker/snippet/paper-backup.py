#date: 2023-06-08T16:43:31Z
#url: https://api.github.com/gists/e729e97116da5543d34d85775709b06b
#owner: https://api.github.com/users/jake-shasteen

# Usage:
# 
# With python3:
# 
# export paperApiKey=<your API key>
# export paperTeamMemberId=<a team member ID>
# 
# python paper-backup.py -d <your backup directory, e.g. ~/paper-backup>
# 
# How do I get those?
# - Go to https://www.dropbox.com/developers and create an app
# - Give it all the read permissions you can
# - Generate an access token (that's your API key!)
# - Go to https://dropbox.github.io/dropbox-api-v2-explorer
# - Switch to business endpoints in the upper right
# - Scroll down the left pane until you find 'groups/list'
# - Use your API key to make a call
# - Find 'everyone at ACS Precon' and copy the group ID
# - Go to 'groups/members/list' and provide the group ID to list members and make the call
# - Scroll and find your name
# - Copy your team_member_id -- use that as 'paperTeamMemberId'
# 
# After running this script, if there are errors in the later stages, you can skip ahead and retry failed folder API calls and failed downloads by providing the manifest file
# which should have been saved in the directory you provided with -d
# Make sure to copy that manifest somewhere else, since that directory gets deleted on startup



import argparse
import json
import os
import requests
import shutil
import sys
import time

def exponentialBackoff(sleepSecs, retries, reqObj, retryFn, onSuccessFn, onFailFn):
    print("%s: (%s) %s -- Retrying with backoff..." % (reqObj.status_code, reqObj.reason, reqObj.text)) 
    while True:
        if retries <= 0:
            print("Too many retries, aborting")
            onFailFn()
            break
        if reqObj.status_code == 200:
            onSuccessFn()
            break
        print("Waiting %s seconds" % sleepSecs)
        time.sleep(sleepSecs)
        sleepSecs = sleepSecs * 2
        retries = retries - 1
        print("%s: (%s) %s -- Retrying request" % (reqObj.status_code, reqObj.reason, reqObj.text))
        retryFn()

def listDocs():
    listUrl = "https://api.dropboxapi.com/2/paper/docs/list"
    listCont = "https://api.dropboxapi.com/2/paper/docs/list/continue"
    headers = {'Authorization': 'Bearer '+apikey,
                    'Dropbox-Api-Select-User': teamMemberId,
                    'Content-Type': 'application/json'}
    listData = {'limit': 100}
    docs = None
    listReq = None

    def retryListReq():
        nonlocal listReq
        listReq = requests.post(listUrl, headers=headers, json=listData)

    def onSuccessListReq():
        print("Got initial list")
        nonlocal docs
        docs = json.loads(listReq.text)

    def onFailListReq():
        sys.exit('error %s' % (listReq.status_code))

    def retryListReqHasMore():
        nonlocal listReq
        listReq = requests.post(listCont, headers=headers, json=cursor)

    def onSuccessListReqHasMore():
        print ("Got more!")
        nonlocal docs
        nonlocal cursor
        nonlocal docList
        docs = json.loads(listReq.text)
        cursor = {'cursor': docs['cursor']['value']}
        docList.extend(docs['doc_ids'])

    def onFailListReqHasMore():
        sys.exit('error %s' % (listReq.status_code))


    retryListReq()

    if listReq.status_code == 200:
        onSuccessListReq()
    else:
        exponentialBackoff(15, 3, listReq, retryListReq, onSuccessListReq, onFailListReq)

    cursor = {'cursor': docs['cursor']['value']}

    docList = docs['doc_ids']

    if docs['has_more'] == True:
        cursorTrue = True
        while cursorTrue:
            if docs['has_more'] == True:
                retryListReqHasMore()
                if listReq.status_code == 200:
                    onSuccessListReqHasMore()
                else:
                    exponentialBackoff(15, 3, listReq, retryListReqHasMore, onSuccessListReqHasMore, onFailListReqHasMore)
            else:
                cursorTrue = False
    else:
        docList = docs['doc_ids']

    return docList


def getFolderInfo(docList = []):
    folderList = []
    failedFolders = []
    folderReq = None
    folderInfo = None
    folderUrl = 'https://api.dropboxapi.com/2/paper/docs/get_folder_info'
    headers = {'Authorization': 'Bearer '+apikey,
                    'Dropbox-Api-Select-User': teamMemberId,
                    'Content-Type': 'application/json'}

    def onRetry():
        nonlocal folderReq
        folderReq = requests.post(folderUrl, headers=headers, json = folderData)

    def onSuccess():
        nonlocal folderInfo
        folderInfo = json.loads(folderReq.text)

    def onFail():
        nonlocal skipFlag
        skipFlag = True

    docListLength = len(docList)
    print('%s docs to process' % (docListLength))
    count = 0
    countTrigger = 200
    for doc in docList:
        count = count + 1
        if count > countTrigger:
            countTrigger += 200
            print('%s of %s docs processed' % (count, docListLength))

        folderData = {'doc_id': doc}
        onRetry()
        
        if folderReq.status_code == 200:
            onSuccess()
        elif folderReq.status_code == 409:
            failedFolders.append(doc)
            continue
        else:
            skipFlag = False
            exponentialBackoff(15, 3, folderReq, onRetry, onSuccess, onFail)
            if skipFlag:
                failedFolders.append(doc)
                continue

        if folderInfo:
            if len(folderInfo['folders']) > 1:
                tree = []
                for folder in folderInfo['folders']:
                    tree.append(folder['name'])
                folderList.append({'docid': doc, 'folderName': '/'.join(tree)})
            else:
                folderList.append({'docid': doc, 'folderName': folderInfo['folders'][0]['name']})
        else:
            folderList.append({'docid': doc, 'folderName': 'root'})

    print("Docs that couldn't be printed %s" % (failedFolders))

    return (failedFolders, folderList)

def downloadAndWrite(folderList):
    url = 'https://api.dropboxapi.com/2/paper/docs/download'
    unwriteable = []
    req = None
    getTitle = None

    def makeRequest():
        nonlocal req
        req = requests.post(url, headers=headers)

    def onSuccess():
        nonlocal getTitle
        getTitle = json.loads(req.headers['Dropbox-Api-Result'])
        pass

    def onFail():
        sys.exit('download error %s' % (req.status_code))
        pass

    for f in folderList:
        headers = {'Authorization': 'Bearer '+apikey,
            'Dropbox-Api-Select-User': teamMemberId,
            'Dropbox-API-Arg': json.dumps({'doc_id': f['docid'], 'export_format': 'markdown'})}
        
        makeRequest()

        if req.status_code == 200:
            onSuccess()
        else:
            exponentialBackoff(15, 3, req, makeRequest, onSuccess, onFail)

        filename = getTitle['title'].replace("/", "").replace(".", "") +'.md'
        foldername = f['folderName']

        try:
            if foldername == 'root':
                combined = os.path.normpath(backup_dir + os.sep)
            else:
                os.makedirs(os.path.normpath(backup_dir)+os.sep+os.path.normpath(foldername), exist_ok=True)
                combined = os.path.normpath(backup_dir+os.sep+os.path.normpath(foldername)+os.sep)
            writefile = open(os.path.join(combined, filename), 'w')
            writefile.write(req.text)
            writefile.close()
            print('saved "%s" to "%s"' % (filename, combined))
        except:
            e = sys.exc_info()[0]
            print("Couldn't write file %s. dest: %s Error %s" % (filename, foldername, e))
            unwriteable.append({ 'doc_id': f['docid'], 'filename': filename, 'foldername': foldername })
            # sys.exit("Couldn't write file.")
    
    errorlog = open(os.path.join(os.path.normpath(backup_dir) + os.sep, 'errors.txt'), 'w')
    errorlog.write(json.dumps(unwriteable))
    errorlog.close()

def saveManifest(failedFolders, folderList):
    # dump backup of folder paths
    try:
        backupFolders = open(os.path.join(os.path.normpath(backup_dir) + os.sep, 'manifest.txt'), 'w')
        backupFolders.write(json.dumps({'failedFolders': failedFolders, 'folderList': folderList}))
        backupFolders.close()
    except:
        print("Could not backup folders")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Export Dropbox Paper Notes.')
    parser.add_argument('-d','--dest', help='Where to store the data.', required=True)
    parser.add_argument('-f','--force', help='Force deletion of destination folder.', required=False, action='store_true')
    parser.add_argument('-m', '--manifest', help='Path to manifest if doc and folder steps already completed.', required=False)
    args = parser.parse_args()


    DEFAULT_SLEEP_SECS = 15

    # Initialize

    try:
        apikey = os.environ['paperApiKey']
        teamMemberId = os.environ['paperTeamMemberId']
    except:
        apikey = ''
        if not apikey:
            sys.exit('Print API Key not set')

    backup_dir = args.dest
    manifest_path = args.manifest
    manifest = None

    if manifest_path is not None:
        try:
            manifest_fd = open(os.path.normpath(manifest_path), 'r')
            manifest = json.loads(manifest_fd.read())
            manifest_fd.close()
        except:
            sys.exit("Could not load manifest")

    if args.force is True:
        try:
            shutil.rmtree(backup_dir)
        except:
            pass
    else:
        if os.path.exists(backup_dir):
            q = input('Destination dir exists already, shall we delete it? (y/n) ')
            if not q == 'y':
                sys.exit('Ok, bye.')
            else:
                try:
                    shutil.rmtree(backup_dir)
                except:
                    sys.exit("Couldn't remove directory")


    if manifest is None:
        docList = listDocs()
        failedFolders, folderList = getFolderInfo(docList)

    if manifest is not None:
        try:
            _folderList = manifest["folderList"]
        except:
            pass

        try:
            docList = manifest["failedFolders"]
        except:
            pass

        failedFolders, folderList = getFolderInfo(docList)
        folderList = folderList + _folderList

    # download / export
    try:
        os.makedirs(os.path.normpath(backup_dir), exist_ok=True)
    except:
        sys.exit("Couldn't create backup folder")

    saveManifest(failedFolders, folderList)

    downloadAndWrite(folderList)

