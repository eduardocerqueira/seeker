#date: 2022-10-24T17:17:21Z
#url: https://api.github.com/gists/439f34605a08ad5e258ad140d02fb186
#owner: https://api.github.com/users/b31ngd3v

import json
from decimal import Decimal

import boto3
import requests


class UpdateDB:
    def __init__(self):
        self.result = []
        self.get_sudoswap()
        self.update_db()

    def update_db(self):
        client_dynamo = boto3.resource("dynamodb")
        table_dynamo = client_dynamo.Table("sudoswap")
        try:
            with table_dynamo.batch_writer() as batch:
                for item in self.result:
                    batch.put_item(Item=item)
        except Exception:
            raise

    def get_sudoswap(self):
        page = 1
        tmp = []
        while True:
            response = requests.get(
                f"https://sudoapi.xyz/v1/collections?sort=volume_all_time&desc=true&pageNumber={page}",
                timeout=10,
            ).json()["collections"]
            if not response:
                break
            res = []
            for i in range(len(response)):
                if response[i]["address"] in tmp:
                    continue
                try:
                    item = {
                        "address": response[i]["address"],
                        "sell_quote": response[i]["sell_quote"],
                        "buy_quote": response[i]["buy_quote"],
                        "collection": response[i],
                    }
                    res.append(json.loads(json.dumps(item), parse_float=Decimal))
                    tmp.append(response[i]["address"])
                except:
                    pass
            self.result += res
            page += 1


def lambda_handler(event, context):
    UpdateDB()
    return "success"
