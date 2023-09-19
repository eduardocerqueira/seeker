#date: 2023-09-19T16:52:01Z
#url: https://api.github.com/gists/5c1baac0a8522252a4a4226298d323ef
#owner: https://api.github.com/users/milov52

def change_state(request, locator):
    order_id = request.GET.get('InvId')
    tg_user_id = Order.objects.filter(id=order_id).first().tg_user_id
    conversation = Conversation.objects.get(tg_user_id=tg_user_id)
    restored_state = restore_state_safely(router, conversation)
    conversation_var.set(conversation)

    state_machine = StateMachine(
        current_state=restored_state or router.locate('/'),
    )

 "**********"  "**********"  "**********"  "**********"  "**********"w "**********"i "**********"t "**********"h "**********"  "**********"S "**********"y "**********"n "**********"c "**********"T "**********"g "**********"C "**********"l "**********"i "**********"e "**********"n "**********"t "**********". "**********"s "**********"e "**********"t "**********"u "**********"p "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"= "**********"s "**********"e "**********"t "**********"t "**********"i "**********"n "**********"g "**********"s "**********". "**********"E "**********"N "**********"V "**********". "**********"T "**********"G "**********". "**********"B "**********"O "**********"T "**********"_ "**********"T "**********"O "**********"K "**********"E "**********"N "**********") "**********": "**********"
        state_machine.switch_to(router.locate(
            locator,
            is_payment=True,
            order_number=order_id
        ))

    Conversation.objects.filter(tg_user_id=tg_user_id).update(
        state_class_locator=locator,
        state_params = {
            "is_payment": True,
            "order_number ": order_id
        }
    )

    if success_payment:
        Order.objects.filter(id=order_id).update(
            paid=True
        )