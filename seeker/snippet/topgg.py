#date: 2022-03-10T17:10:14Z
#url: https://api.github.com/gists/e6d0c771eda1ceedeb0d1ed87dbfe31c
#owner: https://api.github.com/users/martinbndr

import topgg
import aiohttp
# Install the modules with pip install topggpy

class Topgg(commands.Cog):

    def __init__(self, client):
        self.client = client
        self.dbl_token = ""  # Set this to your bot's Top.gg token
        self.discord_webhook = "" # Set this to a discord webhook url
        self.client.topggpy = topgg.DBLClient(client, dbl_token)
        self.client.topgg_webhook = topgg.WebhookManager(client).dbl_webhook("/dblwebhook", "Password") # Set "Password" o a password of <our choice that you add to the webhook settings under Webhooks > Authorization
        self.client.topgg_webhook.run(5000) # Change the Port if needed
        
    @commands.Cog.listener()
    async def on_dbl_vote(self, data):
        userid = data["user"]
        user = await self.client.get_or_fetch_user(int(userid))
        user_created = user.created_at
        async with aiohttp.ClientSession() as session:
            webhook = discord.Webhook.from_url(self.discord_webhook, session=session)
            embed = discord.Embed(title="User Voted", description=f'User: {user}\nID: {user.id}\nUser created: <t:{int(user_created.timestamp())}:F> ( <t:{int(user_created.timestamp())}:R> )', color=discord.Color.blurple())
            embed.set_author(name=user, icon_url=user.avatar.url)
            await webhook.send(embed=embed, avatar_url=self.client.user.avatar.url)

        print(f"Received a vote:\n{data}")


def setup(client):
    client.add_cog(Topgg(client))