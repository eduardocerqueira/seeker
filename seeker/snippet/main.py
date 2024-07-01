#date: 2024-07-01T16:42:07Z
#url: https://api.github.com/gists/6790e773e79943f8634add54e8deba89
#owner: https://api.github.com/users/nuhuh567

import discord
from discord.ext import commands
from discord.utils import utcnow
from datetime import timedelta
import logging
import json
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('discord_bot')

# Load prefix settings from the JSON file
def load_prefix_settings():
    if os.path.exists('data.json'):
        with open('data.json', 'r') as f:
            settings = json.load(f)
    else:
        settings = {}
    return settings

# Function to get the prefix for a given guild
def get_prefix(bot, message):
    settings = load_prefix_settings()
    guild_id = str(message.guild.id)
    return settings.get(guild_id, {}).get('prefix', '-')

# Configure bot with intents
intents = discord.Intents.default()
intents.members = True
intents.guilds = True
intents.messages = True
intents.message_content = True  # Explicitly enable message content

bot = commands.Bot(command_prefix=get_prefix, intents=intents)

# Event when the bot is ready
@bot.event
async def on_ready():
    for filename in os.listdir('./cogs'):
        if filename.endswith('.py'):
            try:
                await bot.load_extension(f'cogs.{filename[:-3]}')
                logger.info(f'Loaded extension: cogs.{filename[:-3]}')
            except Exception as e:
                logger.error(f'Failed to load extension cogs.{filename[:-3]}: {e}')
    logger.info(f'Bot logged in as {bot.user}')
    print(f'Bot logged in as {bot.user}')

    # Sync the commands with Discord
    try:
        await bot.tree.sync()
        logger.info("Slash commands synced successfully.")
    except Exception as e:
        logger.error(f"Error syncing commands: {e}")

# Error handling to log issues
@bot.event
async def on_command_error(ctx, error):
    logger.error(f'Error in command {ctx.command}: {str(error)}')
    if isinstance(error, commands.MissingPermissions):
        embed = discord.Embed(title="Error", description="You don't have permission to use this command.", color=discord.Color.red())
        await ctx.send(embed=embed)
    elif isinstance(error, commands.MemberNotFound):
        embed = discord.Embed(title="Error", description="Member not found.", color=discord.Color.red())
        await ctx.send(embed=embed)
    elif isinstance(error, commands.CommandInvokeError):
        embed = discord.Embed(title="Error", description="There was an error executing the command.", color=discord.Color.red())
        await ctx.send(embed=embed)
    else:
        embed = discord.Embed(title="Error", description=f"An error occurred: {str(error)}", color=discord.Color.red())
        await ctx.send(embed=embed)

# Check if user has the required permissions
def has_permissions(ctx, perm):
    return getattr(ctx.author.guild_permissions, perm)

# Check if the target is valid for moderation action
def can_moderate(ctx, member):
    if member == ctx.author:
        return False, "You cannot moderate yourself."
    if member.top_role >= ctx.author.top_role:
        return False, "You cannot moderate someone with a higher or equal role."
    if member.bot:
        return False, "You cannot moderate a bot."
    return True, ""

# Command to ban a member
@bot.command()
@commands.has_permissions(ban_members=True)
async def ban(ctx, member: discord.Member, *, reason=None):
    logger.debug(f'Attempting to ban {member} with reason: {reason}')
    can_act, msg = can_moderate(ctx, member)
    if not can_act:
        embed = discord.Embed(title="Ban Failed", description=msg, color=discord.Color.red())
        await ctx.send(embed=embed)
        return
    
    if has_permissions(ctx, 'ban_members'):
        try:
            await member.ban(reason=reason)
            embed = discord.Embed(title="Ban", description=f"{member.mention} has been banned.\nReason: {reason}", color=discord.Color.red())
            await ctx.send(embed=embed)
            logger.info(f'Successfully banned {member}')
        except discord.Forbidden:
            embed = discord.Embed(title="Ban Failed", description=f"Cannot ban {member.mention}. Missing permissions.", color=discord.Color.red())
            await ctx.send(embed=embed)
            logger.error(f'Failed to ban {member}. Missing permissions.')
        except Exception as e:
            embed = discord.Embed(title="Ban Failed", description=f"Failed to ban {member.mention}. Error: {str(e)}", color=discord.Color.red())
            await ctx.send(embed=embed)
            logger.error(f'Error banning {member}: {str(e)}')
    else:
        embed = discord.Embed(title="Ban Failed", description="You don't have permission to use this command.", color=discord.Color.red())
        await ctx.send(embed=embed)

# Command to unban a member by user ID
@bot.command()
@commands.has_permissions(ban_members=True)
async def unban(ctx, user_id: int):
    logger.debug(f'Attempting to unban user with ID: {user_id}')
    try:
        user = await bot.fetch_user(user_id)
        await ctx.guild.unban(user)
        embed = discord.Embed(title="Unban", description=f"{user.mention} has been unbanned.", color=discord.Color.green())
        await ctx.send(embed=embed)
        logger.info(f'Successfully unbanned {user}')
    except Exception as e:
        embed = discord.Embed(title="Unban Failed", description=f"Failed to unban user with ID {user_id}. Error: {str(e)}", color=discord.Color.red())
        await ctx.send(embed=embed)
        logger.error(f'Error unbanning user with ID {user_id}: {str(e)}')

# Command to kick a member
@bot.command()
@commands.has_permissions(kick_members=True)
async def kick(ctx, member: discord.Member, *, reason=None):
    logger.debug(f'Attempting to kick {member} with reason: {reason}')
    can_act, msg = can_moderate(ctx, member)
    if not can_act:
        embed = discord.Embed(title="Kick Failed", description=msg, color=discord.Color.red())
        await ctx.send(embed=embed)
        return

    if has_permissions(ctx, 'kick_members'):
        try:
            await member.kick(reason=reason)
            embed = discord.Embed(title="Kick", description=f"{member.mention} has been kicked.\nReason: {reason}", color=discord.Color.orange())
            await ctx.send(embed=embed)
            logger.info(f'Successfully kicked {member}')
        except discord.Forbidden:
            embed = discord.Embed(title="Kick Failed", description=f"Cannot kick {member.mention}. Missing permissions.", color=discord.Color.red())
            await ctx.send(embed=embed)
            logger.error(f'Failed to kick {member}. Missing permissions.')
        except Exception as e:
            embed = discord.Embed(title="Kick Failed", description=f"Failed to kick {member.mention}. Error: {str(e)}", color=discord.Color.red())
            await ctx.send(embed=embed)
            logger.error(f'Error kicking {member}: {str(e)}')
    else:
        embed = discord.Embed(title="Kick Failed", description="You don't have permission to use this command.", color=discord.Color.red())
        await ctx.send(embed=embed)

# Helper function to convert time string to timedelta
def convert_time(time_str):
    try:
        amount = int(time_str[:-1])
        unit = time_str[-1]
        if unit == 's':
            return timedelta(seconds=amount)
        elif unit == 'm':
            return timedelta(minutes=amount)
        elif unit == 'h':
            return timedelta(hours=amount)
        elif unit == 'd':
            return timedelta(days=amount)
        else:
            raise ValueError("Invalid time unit")
    except Exception as e:
        logger.error(f'Error converting time: {str(e)}')
        raise

# Command to timeout (mute) a member
@bot.command(aliases=['mute'])
@commands.has_permissions(moderate_members=True)
async def timeout(ctx, member: discord.Member, time: str, *, reason=None):
    logger.debug(f'Attempting to timeout {member} for {time} with reason: {reason}')
    can_act, msg = can_moderate(ctx, member)
    if not can_act:
        embed = discord.Embed(title="Timeout Failed", description=msg, color=discord.Color.red())
        await ctx.send(embed=embed)
        return

    if has_permissions(ctx, 'moderate_members'):
        try:
            until = utcnow() + convert_time(time)
            await member.timeout(until, reason=reason)
            embed = discord.Embed(title="Timeout", description=f"{member.mention} has been timed out for {time}.\nReason: {reason}", color=discord.Color.blue())
            await ctx.send(embed=embed)
            logger.info(f'Successfully timed out {member} for {time}')
        except discord.Forbidden:
            embed = discord.Embed(title="Timeout Failed", description=f"Cannot timeout {member.mention}. Missing permissions.", color=discord.Color.red())
            await ctx.send(embed=embed)
            logger.error(f'Failed to timeout {member}. Missing permissions.')
        except Exception as e:
            embed = discord.Embed(title="Timeout Failed", description=f"Failed to timeout {member.mention}. Error: {str(e)}", color=discord.Color.red())
            await ctx.send(embed=embed)
            logger.error(f'Error timing out {member}: {str(e)}')
    else:
        embed = discord.Embed(title="Timeout Failed", description="You don't have permission to use this command.", color=discord.Color.red())
        await ctx.send(embed=embed)

# Command to untimeout (unmute) a member
@bot.command(aliases=['unmute'])
@commands.has_permissions(moderate_members=True)
async def untimeout(ctx, member: discord.Member):
    logger.debug(f'Attempting to untimeout {member}')
    if has_permissions(ctx, 'moderate_members'):
        try:
            await member.timeout(None)
            embed = discord.Embed(title="Untimeout", description=f"{member.mention} has been unmuted.", color=discord.Color.green())
            await ctx.send(embed=embed)
            logger.info(f'Successfully unmuted {member}')
        except discord.Forbidden:
            embed = discord.Embed(title="Untimeout Failed", description=f"Cannot untimeout {member.mention}. Missing permissions.", color=discord.Color.red())
            await ctx.send(embed=embed)
            logger.error(f'Failed to untimeout {member}. Missing permissions.')
        except Exception as e:
            embed = discord.Embed(title="Untimeout Failed", description=f"Failed to untimeout {member.mention}. Error: {str(e)}", color=discord.Color.red())
            await ctx.send(embed=embed)
            logger.error(f'Error unmuting {member}: {str(e)}')
    else:
        embed = discord.Embed(title="Untimeout Failed", description="You don't have permission to use this command.", color=discord.Color.red())
        await ctx.send(embed=embed)

# Run the bot with your token
try:
    bot.run('MTIzOTc2NDA3MTc4MDQ1MDMyNA.G88ovT.ykqx1Dof05afpD3vOXK-q6HvqcFNzs7KkozjTs')
except Exception as e:
    logger.critical(f'Error starting bot: {str(e)}')
    print(f'Error starting bot: {str(e)}')
