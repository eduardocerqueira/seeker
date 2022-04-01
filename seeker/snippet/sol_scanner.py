#date: 2022-04-01T16:59:30Z
#url: https://api.github.com/gists/429c92fb0b27b2e21c797d33defd3a41
#owner: https://api.github.com/users/slycaptcha

import asyncio
import json
import logging
from sqlalchemy.orm.session import sessionmaker
import websockets
from typing import List, Sequence, Union
from solana.publickey import PublicKey
from helpers import init_db_session, parse_response
from event import SolanaEvent
from services import hash_response, send_signature_to_relayer, sign_hash


class SolanaEventScanner:
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "logsSubscribe",
    }

    def __init__(self, account_keys: Sequence[Union[str, PublicKey]], **kwargs):
        self.account_keys = account_keys
        self.ws_connection = kwargs["ws_connection"]
        self.commitment = kwargs["commitment"]
        self.db_engine = init_db_session(
            kwargs["db_user"], kwargs["db_password"], kwargs["db_host"],
            kwargs["db_port"], kwargs["db_name"])

    def __set_request(self, public_key: str):
        params: List[dict] = [
            {
                "mentions": [public_key, ]
            },
            {
                "commitment": self.commitment
            }
        ]
        self.request["params"] = params

    def __sign_params(self, parsed_params):
        # Hash parsed response
        hash_packed = hash_response(parsed_params)
        logging.info(f"Hash Packed: {hash_packed}")
        signature = sign_hash(hash_packed)
        return signature

    def __event_to_int(self, parsed_response: dict):
        parsed_response["subscription"] = int(parsed_response["subscription"])
        parsed_response["rbc_amount_in"] = int(
            parsed_response["rbc_amount_in"])
        parsed_response["amount_spent"] = int(parsed_response["amount_spent"])
        parsed_response["token_out_min"] = int(
            parsed_response["token_out_min"])

    def __save_event(self, public_key: str, response):
        logging.info(f"Event: {response}")
        try:
            parsed_response = parse_response(response)
            if not parsed_response:
                return
        except Exception as e:
            logging.warning(
                f"Error parsing response: {e}; Account: {public_key}; Event: {response}")
            return
        logging.info(
            f"Parsed response: {parsed_response}; Account: {public_key}; Event: {response}")
        event = SolanaEvent(public_key=public_key, **parsed_response)
        Session = sessionmaker(self.db_engine)
        with Session() as session:
            try:
                session.add(event)
                session.commit()
            except Exception as e:
                session.rollback()
                logging.warning(
                    f"Session exeption: {e}")
                return
        logging.info(
            f"Signature: {parsed_response.get('signature')}; Account: {public_key}; Subscription: {parsed_response.get('subscription')}")
        self.__event_to_int(parsed_response)
        try:
            signature = self.__sign_params(parsed_response)
        except BaseException as e:
            logging.warning(f"Error get signature: {e}; Event: {response}")
            return
        logging.info(
            f"Signed params: {signature}; Account: {public_key}; Event: {response}")

        send_signature_to_relayer(parsed_response, signature)

    async def __subscribe(self, public_key: str):
        while True:
            try:
                async with websockets.connect(self.ws_connection) as websocket:
                    self.__set_request(public_key)
                    await websocket.send(json.dumps(self.request))
                    first_resp = json.loads(await websocket.recv())
                    logging.info(
                        f"New subscription: {first_resp['result']}; Account: {public_key}")
                    while True:
                        response = await websocket.recv()
                        response_dict = json.loads(response)
                        if response:
                            self.__save_event(public_key, response_dict)
            except Exception as e:
                logging.info(f"Websockets error: {e}")
                continue

    async def start(self):
        coroutines = [self.__subscribe(key)
                      for key in self.account_keys]
        await asyncio.gather(*coroutines)
