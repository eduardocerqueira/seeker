#date: 2022-06-28T17:06:43Z
#url: https://api.github.com/gists/f8f76db9e4aa167b904e01d89de18a08
#owner: https://api.github.com/users/Tishka17

import asyncio
import logging
from operator import itemgetter

from aiogram import Bot, Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters.state import StatesGroup, State

from aiogram_dialog import Dialog, Window, DialogRegistry
from aiogram_dialog.widgets.kbd import ScrollingGroup, Multiselect, Calendar, Radio
from aiogram_dialog.widgets.text import Format, Const


class MySG(StatesGroup):
    s = State()


async def getter(**_kwargs):
    return {
        "users": [(f"user {i}", i) for i in range(1, 100)]
    }


dialog = Dialog(Window(
    Const("Select users"),
    ScrollingGroup(
        Multiselect(
            Format("✓ {item[0]}"),
            Format("{item[0]}"),
            id="ms",
            items="users",
            item_id_getter=itemgetter(1),
        ),
        width=3,
        height=1,
    ),
    Calendar(id="cal"),
    Radio(
        Format("🔘 {item[0]}"), Format("◯ {item[0]}"),
        id="r0",
        items=[("Green", 0), ("Yellow", 1), ("Red", 2)],
        item_id_getter=itemgetter(1),
    ),
    state=MySG.s,
    getter=getter,
))


async def main():
    # real main
    logging.basicConfig(level=logging.INFO)
    storage = MemoryStorage()
    bot = Bot(token=API_TOKEN)
    dp = Dispatcher(bot, storage=storage)
    registry = DialogRegistry(dp)
    registry.register_start_handler(MySG.s)  # resets stack and start dialogs on /start command
    registry.register(dialog)
    await dp.start_polling()


if __name__ == '__main__':
    asyncio.run(main())