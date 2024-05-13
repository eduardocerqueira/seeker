#date: 2024-05-13T16:50:44Z
#url: https://api.github.com/gists/b7e89eb6dc01c5ed637952611cc204a0
#owner: https://api.github.com/users/dev-artisan

# You need to have the 'requests' module installed, see here: https://pypi.org/project/requests/
import requests

# Note regarding 2FA
# You can either disable 'Enforce 2FA' setting and disable '2FA'. Then you can just use your regular user password.
# Or you can just use an app password, e.g. named 'migration' which you can create in 'Personal settings' --> 'Security'. After successful migration you can delete the app password. 
urlFrom = 'https://nextcloud.domainfrom.tld'
authFrom = "**********"

urlTo = 'https://nextcloud.domainto.tld'
authTo = "**********"

# Deck API documentation: https://deck.readthedocs.io/en/latest/API/
# Use API v1.1 with Deck >= 1.3.0
# For Deck >= 1.0.0 and < 1.3.0 change API version in deckApiPath to v1.0 (leave ocsApiPath unchanged)
# Note that exporting / importing attachments only works with API v.1.1
deckApiPath='index.php/apps/deck/api/v1.1'
ocsApiPath='ocs/v2.php/apps/deck/api/v1.0'

headers={'OCS-APIRequest': 'true', 'Content-Type': 'application/json'}
headersOcsJson={'OCS-APIRequest': 'true', 'Accept': 'application/json'}


def getBoards():
    response = requests.get(
            f'{urlFrom}/{deckApiPath}/boards',
            auth=authFrom,
            headers=headers)
    response.raise_for_status()
    return response.json()

def getBoardDetails(boardId):
    response = requests.get(
            f'{urlFrom}/{deckApiPath}/boards/{boardId}',
            auth=authFrom,
            headers=headers)
    response.raise_for_status()
    return response.json()

def getStacks(boardId):
    response = requests.get(
            f'{urlFrom}/{deckApiPath}/boards/{boardId}/stacks',
            auth=authFrom,
            headers=headers)
    response.raise_for_status()
    return response.json()

def getStacksArchived(boardId):
    response = requests.get(
            f'{urlFrom}/{deckApiPath}/boards/{boardId}/stacks/archived',
            auth=authFrom,
            headers=headers)
    response.raise_for_status()
    return response.json()

def getAttachments(boardId, stackId, cardId):
    response = requests.get(
            f'{urlFrom}/{deckApiPath}/boards/{boardId}/stacks/{stackId}/cards/{cardId}/attachments',
            auth=authFrom,
            headers=headers)
    response.raise_for_status()
    return response.json()

def getAttachment(path):
    response = requests.get(
            f'{urlFrom}/{path}',
            auth=authFrom,
            headers=headers)
    response.raise_for_status()
    return response
    
def getComments(cardId):
    response = requests.get(
            f'{urlFrom}/{ocsApiPath}/cards/{cardId}/comments',
            auth=authFrom,
            headers=headersOcsJson)
    response.raise_for_status()
    return response.json()

def createBoard(title, color):
    response = requests.post(
            f'{urlTo}/{deckApiPath}/boards',
            auth=authTo,
            json={
                'title': title,
                'color': color
            },
            headers=headers)
    response.raise_for_status()
    board = response.json()
    boardId = board['id']
    # remove all default labels
    for label in board['labels']:
        labelId = label['id']
        response = requests.delete(
            f'{urlTo}/{deckApiPath}/boards/{boardId}/labels/{labelId}',
            auth=authTo,
            headers=headers)
        response.raise_for_status()
    return board

def createLabel(title, color, boardId):
    response = requests.post(
            f'{urlTo}/{deckApiPath}/boards/{boardId}/labels',
            auth=authTo,
            json={
                'title': title,
                'color': color
            },
            headers=headers)
    response.raise_for_status()
    return response.json()

def createStack(title, order, boardId):
    response = requests.post(
            f'{urlTo}/{deckApiPath}/boards/{boardId}/stacks',
            auth=authTo,
            json={
                'title': title,
                'order': order
            },
            headers=headers)
    response.raise_for_status()
    return response.json()

def createCard(title, ctype, order, description, duedate, boardId, stackId):
    response = requests.post(
            f'{urlTo}/{deckApiPath}/boards/{boardId}/stacks/{stackId}/cards',
            auth=authTo,
            json={
                'title': title,
                'type': ctype,
                'order': order,
                'description': description,
                'duedate': duedate
            },
            headers=headers)
    response.raise_for_status()
    return response.json()

def assignLabel(labelId, cardId, boardId, stackId):
    response = requests.put(
            f'{urlTo}/{deckApiPath}/boards/{boardId}/stacks/{stackId}/cards/{cardId}/assignLabel',
            auth=authTo,
            json={
                'labelId': labelId
            },
            headers=headers)
    response.raise_for_status()

def createAttachment(boardId, stackId, cardId, fileType, fileContent, mimetype, fileName):
    url = f'{urlTo}/{deckApiPath}/boards/{boardId}/stacks/{stackId}/cards/{cardId}/attachments'
    payload = {'type' : fileType}
    files=[
        ('file',(fileName, fileContent, mimetype))
    ]
    response = requests.post( url, auth=authTo, data=payload, files=files)
    response.raise_for_status()
    return response.json()


def createComment(cardId, message):
    response = requests.post(
            f'{urlTo}/{ocsApiPath}/cards/{cardId}/comments',
            auth=authTo,
            json={
                'message': message
            },
            headers=headersOcsJson)
    response.raise_for_status()
    return response.json()

def archiveCard(card, boardId, stackId):
    cardId = card['id']
    card['archived'] = True
    response = requests.put(
            f'{urlTo}/{deckApiPath}/boards/{boardId}/stacks/{stackId}/cards/{cardId}',
            auth=authTo,
            json=card,
            headers=headers)
    response.raise_for_status()

def copyCard(card, boardIdTo, stackIdTo, labelsMap, boardIdFrom):
    createdCard = createCard(
        card['title'],
        card['type'],
        card['order'],
        card['description'],
        card['duedate'],
        boardIdTo,
        stackIdTo
    )

    # copy attachments
    attachments = getAttachments(boardIdFrom, card['stackId'], card['id'])
    for attachment in attachments:
        fileName = attachment['data']
        owner = attachment['createdBy']
        mimetype = attachment['extendedData']['mimetype']
        attachmentPath = attachment['extendedData']['path']
        path = f'remote.php/dav/files/{owner}{attachmentPath}'
        fileContent = getAttachment(path).content
        createAttachment(boardIdTo, stackIdTo, createdCard['id'], attachment['type'], fileContent, mimetype, fileName)

    # copy card labels
    if card['labels']:
        for label in card['labels']:
            assignLabel(labelsMap[label['id']], createdCard['id'], boardIdTo, stackIdTo)

    if card['archived']:
        archiveCard(createdCard, boardIdTo, stackIdTo)

    # copy card comments
    comments = getComments(card['id'])
    if(comments['ocs']['data']):
        for comment in comments['ocs']['data']:
            createComment(createdCard['id'], comment['message'])

def archiveBoard(boardId, title, color):
    response = requests.put(
            f'{urlTo}/{deckApiPath}/boards/{boardId}',
            auth=authTo,
            json={
                'title': title,
                'color': color,
                'archived': True
            },
            headers=headers)
    response.raise_for_status()

# get boards list
print('Starting script')

boards = getBoards()

# create boards
for board in boards:
    boardIdFrom = board['id']
    # create board
    createdBoard = createBoard(board['title'], board['color'])
    boardIdTo = createdBoard['id']
    print('Created board', board['title'])

    # create labels
    boardDetails = getBoardDetails(board['id'])
    labelsMap = {}
    for label in boardDetails['labels']:
        createdLabel = createLabel(label['title'], label['color'], boardIdTo)
        labelsMap[label['id']] = createdLabel['id']

    # copy stacks
    stacks = getStacks(boardIdFrom)
    stacksMap = {}
    for stack in stacks:
        createdStack = createStack(stack['title'], stack['order'], boardIdTo)
        stackIdTo = createdStack['id']
        stacksMap[stack['id']] = stackIdTo
        print('  Created stack', stack['title'])
        # copy cards
        if not 'cards' in stack:
            continue
        for card in stack['cards']:
            copyCard(card, boardIdTo, stackIdTo, labelsMap, boardIdFrom)
        print('    Created', len(stack['cards']), 'cards')

    # copy archived stacks
    stacks = getStacksArchived(boardIdFrom)
    for stack in stacks:
        # copy cards
        if not 'cards' in stack:
            continue
        print('  Stack', stack['title'])
        for card in stack['cards']:
            copyCard(card, boardIdTo, stacksMap[stack['id']], labelsMap, boardIdFrom)
        print('    Created', len(stack['cards']), 'archived cards')

    # archive board if it was archived
    if(board['archived']):
        archiveBoard(board['id'], board['title'], board['color'])
        print('  Archived board') board['title'], board['color'])
        print('  Archived board')