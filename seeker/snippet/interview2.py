#date: 2025-05-28T17:00:30Z
#url: https://api.github.com/gists/94c4def5888687caa4f51a5a57b0bfd9
#owner: https://api.github.com/users/chistyakov

import asyncio

from datetime import datetime

from starlite import Starlite, post
from pydantic import BaseModel

import fake_db

from broker import get_fake_message_broker


class Account(fake_db.Model):
    id = fake_db.IntegerField(primary_key=True, autoincrement=True)
    balance = fake_db.IntegerField()


class Transfer(fake_db.Model):
    id = fake_db.IntegerField(primary_key=True, autoincrement=True)
    sender_id = fake_db.ForeignKey(Account)
    recipient_id = fake_db.ForeignKey(Account)
    amount = fake_db.FloatField()
    created = fake_db.DateTimeField(default=datetime.now)



class PaymentRequest(BaseModel):
    sender_id: int
    recipient_id: int
    amount: float



@post("/payments")
async def makePayment(request: PaymentRequest):
    sender = await Account.get(id=request.sender_id)
    recipient = await Account.get(id=request.recipient_id)

    sender.balance = sender.balance - request.amount
    recipient.balance = recipient.balance + request.amount

    transfer = await Transfer.insert(
        sender_id=request.sender_id,
        recipient_id=request.recipient_id,
        amount=request.amount,
        published_to_broker=False,
    )

    await asyncio.gather(*[sender.save(), recipient.save()])

    broker = get_fake_message_broker()
    await broker.publish(
        {
            "event_type": "transfer_created",
            "event_data": {
                "id": transfer.id,
                "sender_id": request.sender_id,
                "recipient_id": request.recipient_id,
                "amount": request.amount,
                "created": transfer.created,
            },
        }
    )


app = Starlite(route_handlers=[makePayment])
