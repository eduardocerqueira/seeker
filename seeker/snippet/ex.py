#date: 2024-11-26T17:11:38Z
#url: https://api.github.com/gists/074dafb75a7bcd42351344bafe00789d
#owner: https://api.github.com/users/arturboyun

# commands.py
...
commands_router = Router()


@commands_router.message(CommandStart())
@inject
async def command_start_handler(
    message: Message,
    dialog_manager: DialogManager,
    uow: FromDishka[UoW],
    user_service: FromDishka[UserService],
) -> None:
    ...
    logger.debug("Command /start from %s", message.from_user.id)
    await dialog_manager.start(Menu.main, mode=StartMode.RESET_STACK)
...

# ----------------------------------------------------------------------------------------
# main.py
...
dp = Dispatcher(storage=storage)
main_router = Router()
dialogs_router = Router()


async def on_unknown_intent(event: ErrorEvent, dialog_manager: DialogManager):
    # Example of handling UnknownIntent Error and starting new dialog.
    logging.error("Restarting dialog: %s", event.exception)
    if event.update.callback_query:
        await event.update.callback_query.answer(
            "Bot process was restarted due to maintenance.\n"
            "Redirecting to main menu.",
        )
        if event.update.callback_query.message:
            try:
                await event.update.callback_query.message.delete()  # type: ignore
            except TelegramBadRequest:
                pass  # whatever
    elif event.update.message:
        await event.update.message.answer(
            "Bot process was restarted due to maintenance.\n"
            "Redirecting to main menu.",
            reply_markup=ReplyKeyboardRemove(),
        )
    await dialog_manager.start(
        Menu.main,
        mode=StartMode.RESET_STACK,
        show_mode=ShowMode.SEND,
    )


async def start_polling():
    bot = Bot(
        token= "**********"=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )

    main_router.include_router(commands_router)

    dialogs_router.include_routers(
        menu_dialog,
        order_app_dialog,
        admin_menu_dialog,
        admin_add_app_dialog,
        admin_apps_list_dialog,
    )
    main_router.include_router(dialogs_router)
    dp.include_router(main_router)

    dp.errors.register(
        on_unknown_intent,
        ExceptionTypeFilter(UnknownIntent),
    )

    setup_dialogs(dialogs_router)

    container = make_async_container(DepsProvider())
    setup_dishka(container=container, router=dp)

    try:
        await dp.start_polling(bot)
    finally:
        await container.close()
        await bot.session.close()
....close()
...