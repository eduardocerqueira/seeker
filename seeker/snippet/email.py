#date: 2022-10-07T17:17:20Z
#url: https://api.github.com/gists/c06520f8acb307d995b89064f1f46c04
#owner: https://api.github.com/users/bswck

"""
Remember to set the SMTP_HOST environment variable 
or pass it as a keyword argument to EmailWorker()!

Requirements:
pip install flask-mail persist-queue

Usage:
from flask_mail import Message
from email import EmailWorker, start_email_workers, stop_all_workers

john_doe_emails = EmailWorker(
    "john.doe@example.com",
    sender_name="John Doe",
    default_subject="Message from John Doe!",
    storage_path=".john_doe_emails",
    password= "**********"
)

message = Message(
    recipients=["jack.kilby@ti.com"],
    html="Hello, Jack Kilby! Check out my Flask website! https://mywebsite.com/"
)

john_doe_emails.deliver(message)  # notes on the disk to deliver message to Jack 

start_email_workers(app)  # starts john_doe_emails worker thread that will deliver the message
"""

import contextlib
import os
import threading
import time
import traceback
import weakref

import flask
from flask_mail import Mail, Message
from persistqueue import SQLiteAckQueue


class EmailWorker(threading.Thread):
    workers = weakref.WeakSet()

    def __init__(
            self,
            email_address: str | None = None,
            *,
            sender_name: str | None = None,
            default_subject: str = "Message",
            storage_path: str = ".email_cache",
            password: "**********"
            smtp_host: str = None,
            smtp_port: int = 25,
            interval: int = 1,
    ):
        super().__init__()
        self.email_address = email_address
        self.default_subject = default_subject
        self.password = "**********"
        self.interval = interval
        self.sender_name = sender_name or email_address
        self.context = contextlib.nullcontext()

        self.smtp_host = smtp_host or SMTP_HOST
        self.smtp_port = smtp_port or SMTP_PORT

        self.queue = SQLiteAckQueue(storage_path, multithreading=True)
        self.client = Mail()

        self._is_working = False

        EmailWorker.workers.add(self)

    def stop(self):
        self._is_working = False

    def setup(self, app: flask.Flask):
        app.config.update(
            MAIL_SERVER=self.smtp_host,
            MAIL_PORT=self.smtp_port,
            MAIL_USERNAME=self.email_address,
            MAIL_PASSWORD= "**********"
        )
        self.context = app.app_context()
        self.client.init_app(app)

    def error(self, item: dict):
        traceback.print_exc()
        self.queue.ack_failed(id=item["pqid"])

    def send(self, message: Message):
        if not message.sender:
            message.sender = f"{self.sender_name} <{self.email_address}>"
        if not message.subject:
            message.subject = self.default_subject
        self.client.send(message)

    def deliver(self, message: Message):
        return self.queue.put(message)

    def run(self):
        self._is_working = True
        while (
            self._is_working
            and (item := self.queue.get(raw=True))
        ):
            item_id = item["pqid"]
            message = item["data"]
            try:
                with self.context:
                    self.send(message)
            except Exception:
                self.error(item)
            else:
                self.queue.ack(id=item_id)
            time.sleep(self.interval)


def start_email_workers(app):
    workers = set(EmailWorker.workers)

    for worker in workers:
        if not worker.is_alive():
            worker.setup(app)
            worker.start()

    return workers


def stop_all_workers():
    workers = set(EmailWorker.workers)

    for worker in workers:
        worker.stop()
        worker.join()


SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = os.getenv("SMTP_PORT", 25)
