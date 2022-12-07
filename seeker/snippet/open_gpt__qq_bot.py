#date: 2022-12-07T17:01:37Z
#url: https://api.github.com/gists/b5891003ee7bb28de5d617313889468d
#owner: https://api.github.com/users/hhhxiao

import asyncio
import websockets
import json
import requests
import json
key = "**********"
groups = [] #这里填你想让它工作的群的群号，逗号隔开

def send_group_msg(group : int , msg: str):
    return json.dumps({
    "action": "send_msg",
    "params": {
            "message_type": "group",
            "group_id": group,
            "message": msg
        }
    })


def parse_group_msg(msg):
    data = json.loads(msg)
    if 'message_type' in data and 'post_type' in data and 'sender' in data:
        if data['message_type'] == 'group' and data['post_type'] == 'message':
            try:
                msg = data['message']
                sender_number = int(data['sender']['user_id'])
                sender_card = data['sender']['card']
                qq_group_number = int(data['group_id'])
                return True, qq_group_number, sender_number,sender_card,msg
            except:
                return False,None,None,None,None
    return False,None,None,None,None



def launch_post(url,payload=None):
    try:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer {}'.format(key)
        }
        r = requests.post(url,timeout=30,headers=headers,json=payload)
        if r.status_code != 200:
            return False, "发生错误: HTTP状态码为 {}".format(r.status_code)
        return True, r.text
    except Exception as e:
        return False, "发生错误: {}".format(e)

def create_completion(question: str):
    payload = {
        "model": "text-davinci-003",
        "prompt": question,
        "temperature": 0,
        "max_tokens": "**********"
    }

    ok, resp = launch_post('https://api.openai.com/v1/completions',payload)
    if ok:
        try:
            data = json.loads(resp)
            answer = data['choices'][0]['text'].strip()
            return True, answer
        except Exception as e:
            return False, e
    else:
         return False, resp


user_answer_cache = set()
def answer(user: int, question: str):
    if user in user_answer_cache:
        return "用户 {} 的问题尚未回答，请稍后再问".format(user)
    user_answer_cache.add(user)
    ok, ans = create_completion(question)
    user_answer_cache.remove(user)
    return ans


def extract_question_from_msg(msg: str):
    if msg.startswith('Q') or msg.startswith('q'):
        return True, msg[1:]
    else:
        return False,None


async def serve(websocket,path):
    async for msg in websocket:
        ok,group,sender,card,msg = parse_group_msg(msg)
        if ok:
            qok, question = extract_question_from_msg(msg)
            if qok:
                ans = answer(sender,question)
                if len(card.strip()) == 0:
                    card = str(sender) 
                await websocket.send(send_group_msg(group, "回复 {}:\n{}".format(card,ans.strip())))
        else:
            pass



port = 8000
async def main():
    print("Server started on {}".format(port))
    async with websockets.serve(serve, "localhost", port):
        await asyncio.Future()  # run forever

asyncio.run(main())


in())


