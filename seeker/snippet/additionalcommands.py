#date: 2024-07-01T16:42:07Z
#url: https://api.github.com/gists/6790e773e79943f8634add54e8deba89
#owner: https://api.github.com/users/nuhuh567

import discord
from discord.ext import commands, tasks
import logging
import json
import os
import asyncio
from googlesearch import search
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger('discord_bot')

class AdditionalCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.settings = {}
        self.load_data()
        self.check_timeouts.start()
        self.bot.tree.command()(self.help)

    def load_data(self):
        if os.path.exists('data.json'):
            with open('data.json', 'r') as f:
                self.settings = json.load(f)
        else:
            self.settings = {}

    def save_data(self):
        with open('data.json', 'w') as f:
            json.dump(self.settings, f)

    async def help(self, interaction: discord.Interaction):
        await interaction.response.send_message("Select a category:", view=HelpCategoryView(self.bot))

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setlogchannel(self, ctx, channel: discord.TextChannel):
        guild_id = str(ctx.guild.id)
        if guild_id not in self.settings:
            self.settings[guild_id] = {}
        self.settings[guild_id]['log_channel'] = channel.id
        self.save_data()
        embed = discord.Embed(
            description=f"Log channel set to {channel.mention}",
            color=discord.Color.green()
        )
        await ctx.send(embed=embed)

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def prefixset(self, ctx, prefix: str):
        guild_id = str(ctx.guild.id)
        if guild_id not in self.settings:
            self.settings[guild_id] = {}
        self.settings[guild_id]['prefix'] = prefix
        self.save_data()
        embed = discord.Embed(
            description=f"Prefix set to {prefix}",
            color=discord.Color.green()
        )
        await ctx.send(embed=embed)

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def fakepermission(self, ctx, action: str, target: str, permission: str):
        guild_id = str(ctx.guild.id)

        target_member = ctx.guild.get_member_named(target) or ctx.guild.get_member(int(target) if target.isdigit() else 0)
        target_role = discord.utils.get(ctx.guild.roles, name=target) or ctx.guild.get_role(int(target) if target.isdigit() else 0)

        if target_member:
            target_id = str(target_member.id)
            target_mention = target_member.mention
        elif target_role:
            target_id = str(target_role.id)
            target_mention = target_role.mention
        else:
            embed = discord.Embed(
                description="⚠️ Invalid target. Use @user, user ID, @role, or role ID.",
                color=discord.Color.yellow()
            )
            await ctx.send(embed=embed)
            return

        if guild_id not in self.settings:
            self.settings[guild_id] = {}
        if 'permissions' not in self.settings[guild_id]:
            self.settings[guild_id]['permissions'] = {}
        if target_id not in self.settings[guild_id]['permissions']:
            self.settings[guild_id]['permissions'][target_id] = []

        if action.lower() == "grant":
            if permission not in self.settings[guild_id]['permissions'][target_id]:
                self.settings[guild_id]['permissions'][target_id].append(permission)
                embed = discord.Embed(
                    description=f"Granted {permission} to {target_mention}",
                    color=discord.Color.green()
                )
                await ctx.send(embed=embed)
            else:
                embed = discord.Embed(
                    description=f"⚠️ {target_mention} already has {permission} permission",
                    color=discord.Color.yellow()
                )
                await ctx.send(embed=embed)
        elif action.lower() == "remove":
            if permission in self.settings[guild_id]['permissions'][target_id]:
                self.settings[guild_id]['permissions'][target_id].remove(permission)
                embed = discord.Embed(
                    description=f"Revoked {permission} from {target_mention}",
                    color=discord.Color.green()
                )
                await ctx.send(embed=embed)
            else:
                embed = discord.Embed(
                    description=f"⚠️ {target_mention} does not have {permission} permission",
                    color=discord.Color.yellow()
                )
                await ctx.send(embed=embed)
        else:
            embed = discord.Embed(
                description="⚠️ Invalid action. Use 'grant' or 'remove'.",
                color=discord.Color.yellow()
            )
            await ctx.send(embed=embed)

        self.save_data()

    def has_fake_permission(self, ctx, permission):
        guild_id = str(ctx.guild.id)
        user_id = str(ctx.author.id)
        roles_ids = [str(role.id) for role in ctx.author.roles]

        user_permissions = self.settings.get(guild_id, {}).get('permissions', {}).get(user_id, [])
        roles_permissions = [self.settings.get(guild_id, {}).get('permissions', {}).get(role_id, []) for role_id in roles_ids]

        if permission in user_permissions:
            return True
        for role_permissions in roles_permissions:
            if permission in role_permissions:
                return True
        return False

    @commands.command()
    async def ping(self, ctx):
        embed = discord.Embed(
            description=f"Pong! Latency: {round(self.bot.latency * 1000)}ms",
            color=discord.Color.green()
        )
        await ctx.send(embed=embed)

    @commands.command()
    async def warn(self, ctx, member: discord.Member, *, reason=None):
        if reason is None:
            reason = "no reason"
        embed = discord.Embed(
            description=f"{member.mention} You have been warned for breaking the rules, specifically {reason}.",
            color=discord.Color.orange()
        )
        await ctx.send(embed=embed)
        logger.info(f'{member} has been warned by {ctx.author} for {reason}')

        guild_id = str(ctx.guild.id)
        if guild_id in self.settings and 'log_channel' in self.settings[guild_id]:
            log_channel = self.bot.get_channel(self.settings[guild_id]['log_channel'])
            if log_channel:
                embed = discord.Embed(
                    title="Modlog Entry",
                    color=discord.Color.red()
                )
                embed.add_field(name="Information", value="**Case #**: 2252 | Warned", inline=False)
                embed.add_field(name="User", value=f"{member} ({member.id})", inline=False)
                embed.add_field(name="Moderator", value=f"{ctx.author} ({ctx.author.id})", inline=False)
                embed.add_field(name="Reason", value=reason, inline=False)
                embed.set_footer(text=f"{ctx.message.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                await log_channel.send(embed=embed)

    @commands.command()
    async def avatar(self, ctx, member: discord.Member = None):
        member = member or ctx.author
        embed = discord.Embed(
            title=f"{member}'s Avatar",
            color=discord.Color.blue()
        )
        embed.set_image(url=member.avatar.url)
        await ctx.send(embed=embed)

    @commands.command()
    async def userinfo(self, ctx, member: discord.Member = None):
        member = member or ctx.author
        embed = discord.Embed(
            title=f"User Info - {member}",
            color=discord.Color.blue()
        )
        embed.add_field(name="ID", value=member.id, inline=True)
        embed.add_field(name="Name", value=member.display_name, inline=True)
        embed.add_field(name="Account Created", value=member.created_at.strftime('%B %d, %Y'), inline=True)
        embed.add_field(name="Joined Server", value=member.joined_at.strftime('%B %d, %Y'), inline=True)
        embed.set_thumbnail(url=member.avatar.url)
        await ctx.send(embed=embed)

    @commands.command()
    async def serverinfo(self, ctx):
        guild = ctx.guild
        embed = discord.Embed(
            title=f"Server Info - {guild.name}",
            color=discord.Color.blue()
        )
        embed.add_field(name="Server ID", value=guild.id, inline=True)
        embed.add_field(name="Owner", value=guild.owner, inline=True)
        embed.add_field(name="Members", value=guild.member_count, inline=True)
        embed.add_field(name="Roles", value=len(guild.roles), inline=True)
        if guild.icon:
            embed.set_thumbnail(url=guild.icon.url)
        await ctx.send(embed=embed)

    @commands.command(aliases=['r'])
    async def role(self, ctx, action: str, member: discord.Member, role: discord.Role):
        if action.lower() == "add":
            await member.add_roles(role)
            embed = discord.Embed(
                description=f"Added {role.mention} to {member.mention}.",
                color=discord.Color.green()
            )
            await ctx.send(embed=embed)
        elif action.lower() == "remove":
            await member.remove_roles(role)
            embed = discord.Embed(
                description=f"Removed {role.mention} from {member.mention}.",
                color=discord.Color.green()
            )
            await ctx.send(embed=embed)
        else:
            embed = discord.Embed(
                description="⚠️ Invalid action. Use 'add' or 'remove'.",
                color=discord.Color.yellow()
            )
            await ctx.send(embed=embed)

    @commands.command()
    @commands.has_permissions(manage_messages=True)
    async def purge(self, ctx, amount: int, member: discord.Member = None):
        if member:
            msgs = await ctx.channel.history(limit=amount).filter(lambda m: m.author == member).flatten()
            await ctx.channel.delete_messages(msgs)
            embed = discord.Embed(
                description=f"Deleted {len(msgs)} messages from {member.mention}.",
                color=discord.Color.green()
            )
            await ctx.send(embed=embed, delete_after=5)
        else:
            await ctx.channel.purge(limit=amount)
            embed = discord.Embed(
                description=f"Deleted {amount} messages.",
                color=discord.Color.green()
            )
            await ctx.send(embed=embed, delete_after=5)

    @commands.command()
    async def poll(self, ctx, *, question):
        embed = discord.Embed(
            title="Poll",
            description=question,
            color=discord.Color.blue()
        )
        message = await ctx.send(embed=embed)
        await message.add_reaction("👍")
        await message.add_reaction("👎")

    @commands.command()
    async def remindme(self, ctx, time: int, *, task):
        embed = discord.Embed(
            description=f"Reminder set for {time} seconds.",
            color=discord.Color.green()
        )
        await ctx.send(embed=embed)
        await asyncio.sleep(time)
        embed = discord.Embed(
            description=f"{ctx.author.mention}, reminder: {task}",
            color=discord.Color.blue()
        )
        await ctx.send(embed=embed)

    @commands.command()
    async def search(self, ctx, *, query):
        try:
            results = list(search(query, num_results=10))
            pages = [results[i:i+5] for i in range(0, len(results), 5)]

            if not pages:
                embed = discord.Embed(
                    description="⚠️ No results found.",
                    color=discord.Color.yellow()
                )
                await ctx.send(embed=embed)
                return

            current_page = 0

            embed = self.create_search_embed(query, pages, current_page)
            message = await ctx.send(embed=embed)
            await message.add_reaction("⬅️")
            await message.add_reaction("➡️")

            def check(reaction, user):
                return user == ctx.author and str(reaction.emoji) in ["⬅️", "➡️"] and reaction.message.id == message.id

            while True:
                try:
                    reaction, user = await self.bot.wait_for("reaction_add", timeout=60.0, check=check)
                    if str(reaction.emoji) == "⬅️":
                        if current_page > 0:
                            current_page -= 1
                            embed = self.create_search_embed(query, pages, current_page)
                            await message.edit(embed=embed)
                    elif str(reaction.emoji) == "➡️":
                        if current_page < len(pages) - 1:
                            current_page += 1
                            embed = self.create_search_embed(query, pages, current_page)
                            await message.edit(embed=embed)
                    await message.remove_reaction(reaction, user)
                except asyncio.TimeoutError:
                    break
        except Exception as e:
            logger.error(f"Error in command search: {str(e)}")
            embed = discord.Embed(
                description=f"⚠️ An error occurred while searching: {str(e)}",
                color=discord.Color.yellow()
            )
            await ctx.send(embed=embed)

    def create_search_embed(self, query, pages, current_page):
        embed = discord.Embed(
            title=f"Search results for '{query}'",
            color=discord.Color.blue()
        )
        for result in pages[current_page]:
            embed.add_field(name=result, value=f"[Link]({result})", inline=False)
        embed.set_footer(text=f"Page {current_page + 1} of {len(pages)}")
        return embed

    @tasks.loop(seconds=60)
    async def check_timeouts(self):
        now = datetime.utcnow().timestamp()
        for guild_id, settings in self.settings.items():
            if 'timeouts' in settings:
                for timeout in settings['timeouts']:
                    if timeout['end_time'] <= now:
                        guild = self.bot.get_guild(int(guild_id))
                        member = guild.get_member(timeout['member_id'])
                        if member:
                            try:
                                await member.timeout(None)
                                logger.info(f'Automatically unmuted {member} in guild {guild.name}')
                            except Exception as e:
                                logger.error(f'Failed to automatically unmute {member} in guild {guild.name}: {str(e)}')

                self.settings[guild_id]['timeouts'] = [
                    timeout for timeout in settings['timeouts'] if timeout['end_time'] > now
                ]
        self.save_data()

    @check_timeouts.before_loop
    async def before_check_timeouts(self):
        await self.bot.wait_until_ready()

    @commands.command()
    async def custommodmessage(self, ctx, phrase: str, timeout_duration: str, *, response: str):
        allowed_users = [1211795478300336251, 632140453944492032]
        allowed_guilds = [1219848180372606986, 1239750954908389496]
        
        if ctx.author.id not in allowed_users or ctx.guild.id not in allowed_guilds:
            return

        # Validate timeout_duration
        try:
            timeout_duration = int(timeout_duration)
        except ValueError:
            await ctx.send("⚠️ `timeout_duration` must be an integer.")
            return
        
        guild_id = str(ctx.guild.id)
        if guild_id not in self.settings:
            self.settings[guild_id] = {}
        if 'custom_messages' not in self.settings[guild_id]:
            self.settings[guild_id]['custom_messages'] = []

        self.settings[guild_id]['custom_messages'].append({
            'phrase': phrase,
            'response': response,
            'timeout_duration': timeout_duration
        })
        self.save_data()

        embed = discord.Embed(
            description=f"Custom moderation message set for phrase '{phrase}'.",
            color=discord.Color.green()
        )
        await ctx.send(embed=embed)

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author == self.bot.user:
            return

        guild_id = str(message.guild.id)
        if guild_id in self.settings and 'custom_messages' in self.settings[guild_id]:
            for custom_message in self.settings[guild_id]['custom_messages']:
                if custom_message['phrase'].lower() in message.content.lower():
                    await message.channel.send(custom_message['response'])
                    await message.author.timeout(timedelta(seconds=custom_message['timeout_duration']), reason=custom_message['response'])
                    break

class HelpCategoryView(discord.ui.View):
    def __init__(self, bot):
        super().__init__()
        self.bot = bot

        self.add_item(HelpCategoryButton(label="Moderation", style=discord.ButtonStyle.primary, custom_id="help_moderation"))
        self.add_item(HelpCategoryButton(label="Administrators", style=discord.ButtonStyle.danger, custom_id="help_administrators"))
        self.add_item(HelpCategoryButton(label="Base Commands", style=discord.ButtonStyle.secondary, custom_id="help_base"))

class HelpCategoryButton(discord.ui.Button):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def callback(self, interaction: discord.Interaction):
        category_name = self.label
        if category_name == "Moderation":
            commands = [("ban", "Ban a member"), ("kick", "Kick a member"), ("timeout", "Timeout a member"), ("untimeout", "Untimeout a member"), ("warn", "Warn a member"), ("role", "Add or remove a role from a user"), ("purge", "Purge messages from a channel or user")]
        elif category_name == "Administrators":
            commands = [("setlogchannel", "Set the log channel"), ("prefixset", "Set the command prefix"), ("fakepermission", "Grant or remove fake permissions")]
        else:
            commands = [("ping", "Check the bot's latency"), ("avatar", "Show user's avatar"), ("userinfo", "Show user info"), ("serverinfo", "Show server info"), ("poll", "Create a poll"), ("remindme", "Set a reminder"), ("search", "Search the web")]

        embed = discord.Embed(title=f"{category_name} Commands", color=discord.Color.blue())
        page = 1
        pages = (len(commands) - 1) // 5 + 1
        start = (page - 1) * 5
        end = start + 5
        for command, description in commands[start:end]:
            embed.add_field(name=command, value=description, inline=False)
        embed.set_footer(text=f"Page {page}/{pages}")

        await interaction.response.send_message(embed=embed, view=HelpPaginationView(category_name, commands, page, pages))

class HelpPaginationView(discord.ui.View):
    def __init__(self, category_name, commands, page, pages):
        super().__init__()
        self.category_name = category_name
        self.commands = commands
        self.page = page
        self.pages = pages

        self.add_item(HelpPaginationButton(label="Previous", style=discord.ButtonStyle.primary, custom_id="previous_page", page=-1))
        self.add_item(HelpPaginationButton(label="Next", style=discord.ButtonStyle.primary, custom_id="next_page", page=1))

        self.update_buttons()

    def update_buttons(self):
        self.children[0].disabled = self.page <= 1
        self.children[1].disabled = self.page >= self.pages

    async def update_message(self, interaction: discord.Interaction):
        embed = discord.Embed(title=f"{self.category_name} Commands", color=discord.Color.blue())
        start = (self.page - 1) * 5
        end = start + 5
        for command, description in self.commands[start:end]:
            embed.add_field(name=command, value=description, inline=False)
        embed.set_footer(text=f"Page {self.page}/{self.pages}")

        await interaction.response.edit_message(embed=embed, view=self)

class HelpPaginationButton(discord.ui.Button):
    def __init__(self, *args, page, **kwargs):
        super().__init__(*args, **kwargs)
        self.page = page

    async def callback(self, interaction: discord.Interaction):
        view = self.view
        view.page += self.page
        view.update_buttons()
        await view.update_message(interaction)

async def setup(bot):
    await bot.add_cog(AdditionalCommands(bot))
