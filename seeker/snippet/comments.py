#date: 2021-12-16T16:53:04Z
#url: https://api.github.com/gists/ae795093f2139b366418588661e1b218
#owner: https://api.github.com/users/Fom123

from contextlib import suppress
from typing import Tuple

from pyrogram import Client, filters
from pyrogram.errors import MsgIdInvalid
from pyrogram.raw.base import InputPeer
from pyrogram.raw.functions.messages import GetDiscussionMessage
from pyrogram.raw.types import InputPeerChannel
from pyrogram.storage.sqlite_storage import get_input_peer
from pyrogram.types import Message


API_HASH = "a2b483b6f035aee7ba6651ce6c6ffffff" # your hash_id
API_ID = 11111111  # your app_id

app = Client("my_session", api_id=API_ID, api_hash=API_HASH)


async def get_comment_data(
        client: Client,
        peer: InputPeer,
        message: Message
) -> Tuple[InputPeerChannel, int]:
    r = await client.send(GetDiscussionMessage(peer=peer, msg_id=message.message_id))
    m = r.messages[0]
    chat = next(c for c in r.chats if c.id == m.peer_id.channel_id)
    return get_input_peer(chat.id, chat.access_hash, "supergroup"), m.id


@app.on_message(filters.channel & ~filters.edited)
async def send_message(client: Client, message: Message):
    with suppress(MsgIdInvalid):  # if album
        peer, reply_to = await get_comment_data(client, await client.resolve_peer(message.chat.id), message)
        await client.send_message(peer.channel_id, "я бот первонах, вухууу", reply_to_message_id=reply_to)

        
if __name__ == "__main__":
    app.run()