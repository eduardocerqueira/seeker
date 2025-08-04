#date: 2025-08-04T16:49:48Z
#url: https://api.github.com/gists/711164cff67e1b3e27b3bba086ab2a5f
#owner: https://api.github.com/users/Atreus-X

import discord
from discord.ext import commands
import asyncio
import os

# --- Configuration (using environment variables) ---
# Your bot token should be stored in an environment variable named 'DISCORD_BOT_TOKEN'
TOKEN = "**********"
# The channel ID where introductions are posted should be stored in 'INTRO_CHANNEL_ID'
# Make sure to convert it to an integer as IDs are typically integers in discord.py
INTRO_CHANNEL_ID = int(os.environ.get('INTRO_CHANNEL_ID')) 

# --- Discord Bot Setup ---
# We need default intents, plus message_content and members for this bot's functions
# To create and manage channels, the bot will need 'Manage Channels' permission.
intents = discord.Intents.default()
intents.message_content = True  # Required for reading user answers
intents.members = True          # Required for accessing member information

# Initialize the bot. The command_prefix is still needed for !sync command if you want to use it.
bot = commands.Bot(command_prefix='!', intents=intents)

# --- Data Storage ---
# Stores responses temporarily for each user during the introduction process
introduction_responses = {} 

# Stores the temporary channel object for each user
# This allows us to interact with the channel (ask questions, delete it)
temp_channels = {} 

# Store references to the asyncio.Task for each user's timeout so we can cancel them.
temp_channel_timeouts = {} 

# --- Timezone Options (for dropdown menu) ---
TIMEZONE_OPTIONS = [
    "ART (Arabic) Egypt Standard Time) GMT+2:00", 
    "AST (Alaska Standard Time) GMT-9:00", 
    "AGT (Argentina Standard Time) GMT-3:00", 
    "ACT (Australia Central Time) GMT+9:30", 
    "AET (Australia Eastern Time) GMT+10:00", 
    "BST (Bangladesh Standard Time)GMT+6:00", 
    "BET (Brazil Eastern Time) GMT-3:00", 
    "CNT (Canada Newfoundland Time) GMT-3:30", 
    "CAT (Central African Time) GMT-1:00", 
    "CST (Central Standard Time) GMT-6:00", 
    "CTT (China Taiwan Time) GMT+8:00", 
    "EAT (Eastern African Time) GMT+3:00", 
    "EET (Eastern European Time) GMT+2:00", 
    "EST (Eastern Standard Time) GMT-5:00", 
    "ECT (European Central Time) GMT+1:00", 
    "GMT (Greenwich Mean Time) GMT", 
    "HST (Hawaii Standard Time) GMT-10:00", 
    "IST (India Standard Time) GMT+5:30", 
    "IET(Indiana Eastern Standard Time) GMT-5:00", 
    "JST (Japan Standard Time) GMT+9:00", 
    "MET (Middle East Time) GMT+3:30", 
    "MIT (Midway Islands Time) GMT-11:00", 
    "MST (Mountain Standard Time) GMT-7:00", 
    "NET (Near East Time) GMT+4:00", 
    "NST(New Zealand Standard Time) GMT+12:00", 
    "PST (Pacific Standard Time) GMT-8:00", 
    "PLT (Pakistan Lahore Time) GMT+5:00", 
    "PNT (Phoenix Standard Time) GMT-7:00", 
    "PRT (Puerto Rico and US Virgin Islands Time) GMT-4:00", 
    "SST (Solomon Standard Time) GMT+11:00", 
    "UTC (Universal Coordinated Time) GMT", 
    "VST (Vietnam Standard Time) GMT+7:00"
]

# --- Custom UI View for Timezone Selection ---
class TimezoneSelect(discord.ui.Select):
    """Custom dropdown select menu for timezones."""
    def __init__(self, user_id):
        options = []
        for tz in TIMEZONE_OPTIONS:
            options.append(discord.SelectOption(label=tz, value=tz))
        
        super().__init__(placeholder="Choose your timezone...", min_values=1, max_values=1, options=options)
        self.user_id = user_id

    async def callback(self, interaction: discord.Interaction):
        # Store the selected timezone
        introduction_responses[self.user_id]["Timezone"] = self.values[0] # Store the single selected string, the API returns a list
        await interaction.response.send_message(f"You selected: {self.values[0]}", ephemeral=True) # Respond privately
        self.view.stop()  # Stop the view to allow the main command to proceed

# --- Bot Events ---
@bot.event
async def on_ready():
    """Confirms the bot is online and ready and syncs slash commands."""
    print(f'Logged in as {bot.user.name}')
    try:
        await bot.tree.sync() 
        print("Slash commands synced successfully.")
    except Exception as e:
        print(f"Error syncing slash commands: {e}")

# Helper function for cleanup
async def cleanup_introduction(user_id):
    """Cleans up all data and the temporary channel for a user."""
    if user_id in introduction_responses:
        del introduction_responses[user_id]

    if user_id in temp_channel_timeouts:
        temp_channel_timeouts[user_id].cancel()
        del temp_channel_timeouts[user_id]

    if user_id in temp_channels:
        temp_channel = temp_channels[user_id]
        try:
            # Check if the channel still exists before trying to delete
            if bot.get_channel(temp_channel.id):
                await temp_channel.delete(reason="Introduction process completed or cancelled")
        except discord.Forbidden:
            print(f"Warning: Bot lacks permission to delete temporary channel for user {user_id} ({temp_channel.name}).")
            # Attempt to inform the user privately
            user = bot.get_user(user_id)
            if user:
                try:
                    await user.send(f"I've completed your introduction but couldn't delete your temporary channel ({temp_channel.mention}). Please delete it manually if you're done.")
                except discord.Forbidden:
                    print(f"Warning: Could not send DM to user {user_id} about channel deletion failure.")
        except Exception as e:
            print(f"Error deleting temporary channel for user {user_id}: {e}")
            user = bot.get_user(user_id)
            if user:
                try:
                    await user.send(f"An error occurred while deleting your temporary channel ({temp_channel.mention}): {e}. Please delete it manually if you're done.")
                except discord.Forbidden:
                    print(f"Warning: Could not send DM to user {user_id} about channel deletion failure.")
        finally:
            del temp_channels[user_id]

# --- Bot Slash Commands ---
@bot.tree.command(name="introductions", description="Start the introduction process to introduce yourself in a private channel.")
async def introductions_slash(interaction: discord.Interaction):
    """Starts the introduction process for a user, creating a temporary private channel."""
    user = interaction.user
    guild = interaction.guild

    if not guild: # Ensure command is used in a guild
        await interaction.response.send_message("This command can only be used in a server.", ephemeral=True)
        return

    user_id = user.id

    if user_id in introduction_responses:
        await interaction.response.send_message("You are already completing an introduction. Please finish or wait for the process to timeout.", ephemeral=True)
        return

    # Defer the initial response (ephemeral)
    await interaction.response.defer(ephemeral=True, thinking=True) 
    
    # --- Create Temporary Channel ---
    channel_name = f"intro-{user.name.lower().replace(' ', '-')}-{user.discriminator}"
    channel_name = "".join(c for c in channel_name if c.isalnum() or c == '-').lower() 

    overwrites = {
        guild.default_role: discord.PermissionOverwrite(read_messages=False),
        user: discord.PermissionOverwrite(read_messages=True, send_messages=True),
        guild.me: discord.PermissionOverwrite(read_messages=True, send_messages=True, manage_channels=True)
    }

    try:
        temp_channel = await guild.create_text_channel(
            channel_name, 
            overwrites=overwrites,
            reason=f"Introduction channel for {user.display_name}"
        )
        temp_channels[user_id] = temp_channel # Store the channel object
        introduction_responses[user_id] = {} # Initialize responses for this user

        await interaction.followup.send(
            f"I've created a private channel for your introduction: {temp_channel.mention}. "
            "Please go there to answer the questions. You can type 'quit' or 'restart' at any time.", 
            ephemeral=True
        )
        # Send first message in the new private channel
        await temp_channel.send(f"Hello {user.mention}! Please answer the following questions to introduce yourself.")

    except discord.Forbidden:
        await interaction.followup.send("I don't have permissions to create channels. Please check my role permissions.", ephemeral=True)
        await cleanup_introduction(user_id) # Clean up partial state
        return
    except Exception as e:
        await interaction.followup.send(f"An error occurred while creating the channel: {e}", ephemeral=True)
        await cleanup_introduction(user_id) # Clean up partial state
        return
    
    # Start the timeout task
    async def channel_timeout():
        await asyncio.sleep(600) # 10 minutes timeout for the entire process
        if user_id in temp_channels: # Only proceed if still active
            await temp_channels[user_id].send(f"{user.mention}, you took too long to complete your introduction. The process has been cancelled.")
            await cleanup_introduction(user_id)

    timeout_task = bot.loop.create_task(channel_timeout())
    temp_channel_timeouts[user_id] = timeout_task

    # --- Questions and Answers ---
    questions = [
        "What is your real name? (Optional)",
        "Where are you located?",
        "What server/alliance are you coming from?",
        "What is your native language?",
        "Do you speak any others?"
    ]

    for question in questions:
        await temp_channel.send(question) # Ask in the temporary channel
        try:
            # Wait for message in the temporary channel from the user
            message = await bot.wait_for(
                'message', 
                check=lambda m: m.author == user and m.channel == temp_channel, 
                timeout=60.0 # Timeout for each individual question
            )
            response_text = message.content.strip().lower()

            if response_text == 'quit':
                await temp_channel.send("Introduction process cancelled.")
                await cleanup_introduction(user_id)
                return # Exit the command
            elif response_text == 'restart':
                await temp_channel.send("Restarting introduction process...")
                await cleanup_introduction(user_id)
                # Re-invoke the command to restart (this will create a new channel)
                await introductions_slash(interaction) 
                return # Exit the current command instance
            
            introduction_responses[user_id][question] = message.content # Store original case
            # Removed the bot repeating the user's answer based on previous feedback

        except asyncio.TimeoutError:
            await temp_channel.send(f"{user.mention}, you took too long to respond to the last question. The process has been cancelled.")
            await cleanup_introduction(user_id)
            return
        except asyncio.CancelledError:
            # This can happen if the main timeout task cancels the whole interaction.
            # No need to send message here as the timeout task will handle it.
            return 

 # --- Timezone Question (Dropdown) ---
    await temp_channel.send("Please select your timezone:")
    
    view = discord.ui.View(timeout=180) # Give more time for selection if needed
    view.add_item(TimezoneSelect(user_id))
    
    # Send the message with the dropdown menu in the temporary channel
    await temp_channel.send("Choose your timezone from the dropdown:", view=view)
    
    try:
        await view.wait()  # This waits until view.stop() is called in the callback or it times out
        
        # After selection, the callback runs and view.stop() is called.
        # Now check if the user had quit/restart before selecting dropdown
        # This is a fallback check in case the user types before selecting.
        # However, with a View, the primary interaction is the selection.

        if "Timezone" not in introduction_responses[user_id]:
            # This case might be hit if the view times out
            await temp_channel.send(f"{user.mention}, you took too long to select your timezone. The process has been cancelled.")
            await cleanup_introduction(user_id)
            return

    except asyncio.TimeoutError:
        await temp_channel.send(f"{user.mention}, you took too long to select your timezone. The process has been cancelled.")
        await cleanup_introduction(user_id)
        return
    except asyncio.CancelledError:
        # This happens if the main timeout or a quit/restart cancels the interaction.
        return 
    finally:
        # Stop the timeout task once the interaction is completed or failed for this user
        if user_id in temp_channel_timeouts:
            temp_channel_timeouts[user_id].cancel()
            del temp_channel_timeouts[user_id]

    # --- Compile and Post Introduction ---
    introduction_message = f"**New Introduction from {user.display_name} (ID: {user.id}):**\n"
    for question, answer in introduction_responses[user_id].items():
        # Clean up the question string for the final output
        cleaned_question = question.replace('(Optional)', '').strip()
        introduction_message += f"**{cleaned_question}:** {answer}\n"

    target_channel = bot.get_channel(INTRO_CHANNEL_ID)
    if target_channel:
        try:
            await target_channel.send(introduction_message) # This message is public
            await temp_channel.send("Your introduction has been posted to the introductions channel!")
        except discord.Forbidden:
            await temp_channel.send(f"Error: I don't have permissions to post in the designated introduction channel ({target_channel.mention}). Please check my permissions in that channel.", ephemeral=False)
            print(f"Error: Bot lacks permissions to post in channel ID {INTRO_CHANNEL_ID}")
        except Exception as e:
            await temp_channel.send(f"An unexpected error occurred while posting your introduction: {e}", ephemeral=False)
            print(f"Error posting introduction for {user.id}: {e}")
    else:
        await temp_channel.send("Error: Could not find the introduction channel. Please contact an admin to ensure `INTRO_CHANNEL_ID` is correct.", ephemeral=False)
    
    # --- Cleanup ---
    await cleanup_introduction(user_id) # Use the helper function for final cleanup

# --- Run the Bot ---
 "**********"i "**********"f "**********"  "**********"T "**********"O "**********"K "**********"E "**********"N "**********"  "**********"i "**********"s "**********"  "**********"N "**********"o "**********"n "**********"e "**********": "**********"
    print("Error: "**********"
    print("Please set the 'DISCORD_BOT_TOKEN' environment variable with your bot token.")
elif INTRO_CHANNEL_ID == 0: # Assuming 0 is an invalid default or unset state
    print("Error: INTRO_CHANNEL_ID environment variable not set or invalid.")
    print("Please set the 'INTRO_CHANNEL_ID' environment variable with the target channel ID.")
else:
    bot.run(TOKEN)
