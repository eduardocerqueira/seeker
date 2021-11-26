#date: 2021-11-26T17:09:43Z
#url: https://api.github.com/gists/505e760049afaef6ac2a2f31158ce6e9
#owner: https://api.github.com/users/nerdguyahmad

# Install Neocord (git required):
# pip install git+github.com/nerdguyahmad/neocord.git

import neocord
import random
import asyncio
import logging

client = neocord.Client(intents=neocord.GatewayIntents.all())

# you can omit this if you don't want logs.
logging.basicConfig(level=logging.INFO)

# "once" kwarg makes this listener temporary and will only
# call it once.
@client.on('ready', once=True)
async def on_ready():
    print(f'{client.user} is ready.')

# message (command) listener.
@client.on('message')
async def on_message(message):
    if message.author.bot:
        # don't respond to ourself.
        return

    content = message.content.lower()

    if content == '!guess':
        _check = lambda m: all([
                            m.channel.id == message.channel.id,
                            m.author.id == message.author.id,
                            m.content.isdigit()
                        ])

        number = random.randint(1, 20)
        initial_message = await message.channel.send('Guess the number! The number is between 1 to 20. You have got 5 attempts.')
        guess = None
        for attempt in range(5):
            try:
                guess_msg = await client.wait_for('message', check=_check, timeout=90)
            except asyncio.TimeoutError:
                return await message.channel.send('You took too long to respond. :(')
            else:
                guess = int(guess_msg.content) # type: ignore

                if guess == number:
                    return await message.channel.send('Right guess! You won! :tada:')
                if guess > number:
                    await message.channel.send('Your guess is greater then number.')
                elif guess < number:
                    await message.channel.send('Your guess is lower then number.')
        if guess is None:
          await message.channel.send('You ran out of attempts. :(')

client.run('[token]')