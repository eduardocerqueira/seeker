#date: 2023-02-06T17:07:09Z
#url: https://api.github.com/gists/d69e2cc2e6f738f22e79fa67282cb66a
#owner: https://api.github.com/users/vlkorsakov

dialog = Dialog(
    Window(
        Const("Записи:"),
        Format("{records}"),
        Row(
            Button(Const("<"), id="prev_records", on_click=prev_records),
            Button(Const(">"), id="next_records", on_click=next_records),
        ),
        Cancel(Const("Close btn")),
        state=RecordsSG.SHOW,
        getter=get_records,
    ),
)


async def prev_records(_, __, manager: DialogManager):
    offset = manager.dialog_data.get("offset")
    manager.dialog_data.update(offset=offset - 5)


async def next_records(_, __, manager: DialogManager):
    offset = manager.dialog_data.get("offset")
    manager.dialog_data.update(offset=offset - 5)


async def get_records(dialog_manager: DialogManager, repo: Repo, **_):
    offset = dialog_manager.dialog_data.get("offset")
    records = await repo.get_records(offset=offset, limit=5)
    return {
        "records": "\n".join(records)
    }
