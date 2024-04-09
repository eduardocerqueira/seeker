#date: 2024-04-09T16:52:35Z
#url: https://api.github.com/gists/07e679cf8b713a69cc2b943431a058b7
#owner: https://api.github.com/users/umanaww

import asyncio
from contextlib import suppress
import re
from datetime import datetime, timedelta
from aiogram import Bot, Dispatcher, Router, F
from aiogram.enums import ParseMode
from aiogram.client.bot import DefaultBotProperties
from aiogram.filters import Command, CommandObject
from aiogram.types import Message, ChatPermissions
from aiogram.exceptions import TelegramBadRequest

TOKEN = "7150105554: "**********"
GROUPS_ID = [
    -1002011503043,
    -1002101820465,
    935729670,
    1723892685
]
BAD_WORDS = [
    'потом добавим'
]

router = Router()


@router.message(Command("ban"))
async def ban(message: Message, bot: Bot, command: CommandObject | None = None):
    if message.from_user.id == GROUPS_ID[-1] or message.from_user.id == GROUPS_ID[-2]:
        reply_msg = message.reply_to_message
        if not reply_msg:
            return None

        date = parse_time(command.args)
        mention = reply_msg.from_user.mention_html(reply_msg.from_user.first_name)

        with suppress(TelegramBadRequest):
            await bot.ban_chat_member(
                chat_id=message.chat.id,
                user_id=reply_msg.from_user.id,
                until_date=date
            )

        await message.answer(f"Бан лошку @{(reply_msg.from_user.username)}")
    else:
        await message.reply("У тебя нет прав!")


def parse_time(time: str | None):
    if not time:
        return None

    match_ = re.match(r'(\d+)([smhdw])', time.lower().strip())
    cur_date = datetime.now()

    if match_:
        value, unit = int(match_.group(1)), match_.group(2)

        match unit:
            case "s":
                time_delta = timedelta(seconds=value)
            case "m":
                time_delta = timedelta(minutes=value)
            case "h":
                time_delta = timedelta(hours=value)
            case "d":
                time_delta = timedelta(days=value)
            case "w":
                time_delta = timedelta(weeks=value)
            case _:
                return None
    else:
        return None

    new_date = cur_date + time_delta
    unix_timestamp = int(new_date.timestamp())
    return unix_timestamp


async def unmute_message_after_delay(bot, chat_id, user_id, username, delay=300):
    await asyncio.sleep(delay)  # ждем указанное время
    await bot.restrict_chat_member(
        chat_id=chat_id,
        user_id=user_id,
        permissions=ChatPermissions(
            can_send_messages=True
        )
    )
    await bot.send_message(chat_id, f"Пользователь @{username} размучен")


@router.message(Command("mute"))
async def mute(message: Message, bot: Bot, command: CommandObject | None = None):
    if message.from_user.id == GROUPS_ID[-1] or message.from_user.id == GROUPS_ID[-2]:
        reply_msg = message.reply_to_message
        if not reply_msg:
            return None

        time_str = message.text.split(" ")[1]
        date = parse_time(time_str)
        mention = reply_msg.from_user.mention_html(reply_msg.from_user.first_name)

        with suppress(TelegramBadRequest):
            await bot.restrict_chat_member(
                chat_id=message.chat.id,
                user_id=reply_msg.from_user.id,
                until_date=date,
                permissions=ChatPermissions(
                    can_send_messages=False
                )
            )

        await message.answer(f"Замучен @{(reply_msg.from_user.username)}")
        delay = int(date - datetime.now().timestamp())
        asyncio.create_task(
            unmute_message_after_delay(bot, message.chat.id, reply_msg.from_user.id, reply_msg.from_user.username,
                                       delay))
    else:
        await message.reply("У тебя нет прав!")


@router.message(Command("unmute"))
async def unmute(message: Message, bot: Bot, command: CommandObject | None = None):
    if message.from_user.id == GROUPS_ID[-1] or message.from_user.id == GROUPS_ID[-2]:
        reply_msg = message.reply_to_message
        if not reply_msg:
            return None

        date = parse_time(command.args)
        mention = reply_msg.from_user.mention_html(reply_msg.from_user.first_name)

        with suppress(TelegramBadRequest):
            await bot.restrict_chat_member(
                chat_id=message.chat.id,
                user_id=reply_msg.from_user.id,
                until_date=date,
                permissions=ChatPermissions(
                    can_send_messages=True
                )
            )

        await message.answer(f"Размучен @{(reply_msg.from_user.username)}")
    else:
        await message.reply("У тебя нет прав!")


async def delete_message_after_delay(message, delay=300):
    await asyncio.sleep(delay)
    await message.delete()


@router.message(F.text)
async def profanity_filter(message: Message, bot: Bot):
    if any(word in str(message.text).lower() for word in BAD_WORDS):
        asyncio.create_task(delete_message_after_delay(message))


async def main() -> None:
    bot = "**********"=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()

    dp.include_router(router)

    await bot.delete_webhook(True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
__":
    asyncio.run(main())
