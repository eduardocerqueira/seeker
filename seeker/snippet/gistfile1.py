#date: 2023-11-03T16:42:28Z
#url: https://api.github.com/gists/5a54c800198134408ccf075a6766de08
#owner: https://api.github.com/users/Jourdelune

class PatreonApi:
    def __init__(self, campaign_id: "**********": str):
        self.campaign_id = campaign_id
        self.access_token = "**********"

    async def fetch_all(self) -> dict:
        """Get all discord id with current tiers of a patreon campaign

        :return: a dict that contain the discord id and the current tiers of the user
        :rtype: dict
        """        
        url = f'https://www.patreon.com/api/oauth2/v2/campaigns/{self.campaign_id}/members'
        params = {
            'include': 'user,currently_entitled_tiers',
            'fields[user]': 'social_connections',
        }

        headers = {'Authorization': "**********"
        patreons = {}

        end_cursor = False
        while not end_cursor:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    patreon_data = await response.json()

                    for data in patreon_data['data']:
                        discord_user_id = None 
                        for patreon_user in patreon_data['included']:
                            if patreon_user['type'] != 'user':
                                continue
                            
                            if data['relationships']['user']['data']['id'] != patreon_user['id']:
                                continue 

                            patreon_user_data = patreon_user['attributes']
                            if not 'social_connections' in patreon_user_data:
                                continue

                            discord_data = patreon_user['attributes']['social_connections']['discord']
                        
                            if not discord_data or not 'user_id' in discord_data:
                                continue
          
                            discord_user_id = int(discord_data['user_id'])
                           
                        patreons[data['relationships']['user']['data']['id']] = { 
                            'tiers': [d['id'] for d in data['relationships']['currently_entitled_tiers']['data']],
                            'discord': discord_user_id
                        }

                    pagination_data = patreon_data['meta']['pagination']
                    if not pagination_data.get('cursors') or pagination_data['cursors']['next'] == None:
                        end_cursor = True
                    else:
                        next_cursor_id = pagination_data['cursors']['next']
                        params['page[cursor]'] = next_cursor_id

        return patreons

    async def fetch(self, discord_id: int) -> dict:
        """Get all current tiers of a discord user

        :param discord_id: the discord id of the user
        :type discord_id: int
        :return: all current tiers of the user
        :rtype: dict
        """        
        patreon = {}

        url = f'https://www.patreon.com/api/oauth2/v2/campaigns/{self.campaign_id}/members'
        params = {
            'include': 'user,currently_entitled_tiers,pledge_history',
            'fields[user]': 'social_connections',
        }
        headers = {'Authorization': "**********"

        end_cursor = False
        while not end_cursor:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    patreon_data = await response.json()
                    for data in patreon_data['data']:
                        patreon_user_id = data['relationships']['user']['data']['id']

                        for patreon_user in patreon_data['included']:
                            if patreon_user.get('type') != 'user':
                                continue
                            
                            if patreon_user_id != patreon_user['id']:
                                continue 

                            patreon_user_data = patreon_user['attributes']
                            if not patreon_user_data.get('social_connections'):
                                continue

                            discord_data = patreon_user['attributes']['social_connections']['discord']
                        
                            if not discord_data:
                                continue

                            discord_user_id = int(discord_data['user_id'])
                            if discord_user_id != discord_id:
                                continue
                            
                            return {'discord': discord_user_id, 'patreon': str(patreon_user_id), 'tiers': [tier['id'] for tier in data['relationships']['currently_entitled_tiers']['data']]}

                    pagination_data = patreon_data['meta']['pagination']
                    if not pagination_data.get('cursors') or pagination_data['cursors']['next'] == None:
                        end_cursor = True
                    else:
                        next_cursor_id = pagination_data['cursors']['next']
                        params['page[cursor]'] = next_cursor_id

# usage example
api = "**********"
all_patreons = asyncio.run(api.fetch_all())
mpaign_id', 'access_token')
all_patreons = asyncio.run(api.fetch_all())
