#date: 2023-10-23T16:56:32Z
#url: https://api.github.com/gists/eeaf0d45ba9edad855f821e0219cffac
#owner: https://api.github.com/users/NomadWithoutAHome

import disnake
from disnake.ext import commands
import requests
import json
import re
from fuzzywuzzy import fuzz
from disnake.ext.commands.errors import CommandOnCooldown

class EpisodeSearchCog(commands.Cog):
    def __init__(self, bot):
        self.episode_data = None
        self.bot = bot
        self.data_url = "https://www.nobss.online/static/data/episode_data.json"
        self.load_episode_data()

    def load_episode_data(self):
        # Download and load the JSON data
        response = requests.get(self.data_url)
        if response.status_code == 200:
            self.episode_data = json.loads(response.text)
        else:
            self.episode_data = {}

    @commands.slash_command(
        name="searchepisode",
        description="Search for specific episodes of the show",
    )
    @commands.cooldown(1, 60, commands.BucketType.user)
    async def search_episode(self, ctx: disnake.ApplicationCommandInteraction, query: str, fuzzy: bool = False,
                             partial: bool = False, pattern: bool = False):
        try:
            # Search for episodes that match the query
            matching_episodes = self.find_matching_episodes(query, fuzzy, partial, pattern)

            if not matching_episodes:
                await ctx.send("No matching episodes found.")
                return

            embed_list = []  # A list to store the episode embeds

            for episode in matching_episodes[:5]:
                embed = self.create_embed(episode)
                embed_list.append(embed)

            if embed_list:
                # Send all the episode embeds together in one message
                await ctx.send(embeds=embed_list)
        except commands.CommandOnCooldown as e:
            # The user is on cooldown; inform them about the remaining cooldown time
            remaining = round(e.retry_after)
            await ctx.send(f"This command is on cooldown. You can use it again in {remaining} seconds.")

    def find_matching_episodes(self, query, fuzzy, partial, pattern):
        matching_episodes = []

        for season_name, episodes in self.episode_data.items():
            for episode in episodes:
                episode_title = episode['Episode Title'].lower()

                if fuzzy:
                    fuzz_ratio = fuzz.ratio(query.lower(), episode_title)
                    if fuzz_ratio >= 90:
                        matching_episodes.append(episode)
                elif partial:
                    if query.lower() in episode_title:
                        matching_episodes.append(episode)
                else:
                    pattern = r'\b{}\b'.format(re.escape(query.lower()))
                    if re.search(pattern, episode_title):
                        matching_episodes.append(episode)

        return matching_episodes

    def create_embed(self, episode):
        embed = disnake.Embed(
            title=f"{episode['Episode Title']}",
            description=episode["Episode Description"],
            color=disnake.Color.green(),
        )
        embed.add_field(name="Episode Number", value=episode["Episode Number"])
        # Update the key for season number
        season_number = episode.get("Season Number", "N/A")
        embed.add_field(name="Season Number", value=season_number)
        embed.add_field(name="Air Date", value=episode["Air Date"])
        embed.add_field(name="Runtime", value=episode["Runtime"])
        embed.add_field(name="Watch", value=f"[Watch Episode](https://www.nobss.online/video/{episode['uuid']})")
        embed.set_thumbnail(url=episode["Thumbnail"])

        return embed

def setup(bot):
    bot.add_cog(EpisodeSearchCog(bot))
