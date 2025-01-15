#date: 2025-01-15T16:51:24Z
#url: https://api.github.com/gists/c25893826c32cf45e8b09a8f54392a00
#owner: https://api.github.com/users/loxlikooy

@dp.message(F.text)
async def on_message(message: Message):
    res = await find_original_post(
        message.chat.id, message.reply_to_message.message_id
    )
    if res and res.caption:
        name = f'{message.from_user.full_name} | {message.from_user.username}'
        if [f'{message.from_user.id}'] not in worksheet0.get_values('A1:A10000'):
            row = await find_first_empty_row(worksheet0)
            row2 = row + 1
            cell1 = rowcol_to_a1(row, col=6)
            cell2 = rowcol_to_a1(row, col=7)
            values1 = ([
                f'=HYPERLINK("https://t.me/{message.from_user.username}", "{message.from_user.id}")\n\n',
                f'{name}',
                f'{res.caption.splitlines()[0]}',
                f'{f"https://t.me/c/{str(res.chat.id)[4:]}/{res.id}"}',
                f'{message.text}',
                '',
                '',
                f'=ЗНАЧЕН(ПОДСТАВИТЬ({cell1}, ",", ".")) * ЗНАЧЕН(ПОДСТАВИТЬ({cell2}, ",", "."))',
            ])
            values2 = ([
                f'Итог {name}: {message.from_user.id}',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
            ])
            formatcell1 = rowcol_to_a1(row, 1)
            formatcell2 = rowcol_to_a1(row, 8)
            formatcell3 = rowcol_to_a1(row2, 1)
            formatcell4 = rowcol_to_a1(row2, 8)
            values = [values1] + [values2]
            updates_main = [{'range': f'{formatcell1}:{formatcell4}', 'values': values}]
            format_main = [{'range': f'{formatcell1}:{formatcell2}', 'format': format0},
                           {'range': f'{formatcell3}:{formatcell4}', 'format': format1}]
            worksheet0.batch_update(updates_main, value_input_option=user_entered)
            worksheet0.batch_format(format_main)
        else:
            row = worksheet0.get_values('A1:A10000').index([f'{message.from_user.id}']) + 2
            cell1 = rowcol_to_a1(row, col=6)
            cell2 = rowcol_to_a1(row, col=7)
            values1 = [
                '',
                '',
                f'{res.caption.splitlines()[0]}',
                f'{f"https://t.me/c/{str(res.chat.id)[4:]}/{res.id}"}',
                f'{message.text}',
                '',
                '',
                f'=ЗНАЧЕН(ПОДСТАВИТЬ({cell1}, ",", ".")) * ЗНАЧЕН(ПОДСТАВИТЬ({cell2}, ",", "."))',
            ]
            worksheet0.insert_row(
                values1, index=row, inherit_from_before=True,
                value_input_option=user_entered
            )
