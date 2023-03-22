#date: 2023-03-22T16:59:53Z
#url: https://api.github.com/gists/887d2fd942f1c2979e92da095a5d39c6
#owner: https://api.github.com/users/brooney-som

async def update_message(self, ts, channel, text):
        try:
            response = await self.client.chat_update(ts=ts, channel=channel,

                blocks=my_blocks.get_offer_block(username="bill.rooney", instance_id=1, instance_type=1,
                                                 instance_state=1, region=1, session_id=1, idle_timeout=1, extend_by=1))
            return response
        except SlackApiError as e:
            assert e.response["ok"] is False
            assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
            print(f"Got an error: {e.response['error']}")

    async def update_offer(self, ts, channel):
        try:
            response = await self.client.chat_update(ts=ts, channel=channel, text="Updated offer",
                blocks=my_blocks.get_update_offer_block(username="bill.rooney", instance_id=9, instance_type=9,
                                                        instance_state=9, region=9, session_id=9, idle_timeout=9,
                                                        extend_by=9, reason="Unable to extend session"))
            return response
        except SlackApiError as e:
            assert e.response["ok"] is False
            assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
            print(f"Got an error: {e.response['error']}")
