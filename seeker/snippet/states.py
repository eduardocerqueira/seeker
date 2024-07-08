#date: 2024-07-08T16:53:14Z
#url: https://api.github.com/gists/8d3073c61a5347945f9fb6fd9f0bba44
#owner: https://api.github.com/users/coder2020official

import telebot

from telebot import custom_filters, types
from telebot.states import State, StatesGroup #States
from telebot.states.sync.context import StateContext

from telebot.storage import StateMemoryStorage


state_storage = StateMemoryStorage()
bot = "**********"
state_storage=state_storage, use_class_middlewares=True)



class MyStates(StatesGroup):
    name = State()
    surname = State()
    age = State()


@bot.message_handler(commands=['start'])
def start_ex(message: types.Message, state_context: StateContext):
    state_context.set(MyStates.name)
    bot.send_message(message.chat.id, 'Hi, write me a name', reply_to_message_id=message.message_id)
 

# Any state
@bot.message_handler(state="*", commands=['cancel'])
def any_state(message: types.Message, state_context: StateContext):
    state_context.delete(MyStates.name)
    bot.send_message(message.chat.id, 'State has been deleted', reply_to_message_id=message.message_id)

@bot.message_handler(state=MyStates.name)
def name_get(message: types.Message, state_context: StateContext):
    state_context.set(MyStates.surname)
    bot.send_message(message.chat.id, 'Now write me a surname', reply_to_message_id=message.message_id)
    state_context.add_data(name=message.text)
 
 
@bot.message_handler(state=MyStates.surname)
def ask_age(message: types.Message, state_context: StateContext):
    bot.send_message(message.chat.id, "What is your age?", reply_to_message_id=message.message_id)
    state_context.set(MyStates.age)
    state_context.add_data(surname=message.text)
 

@bot.message_handler(state=MyStates.age, is_digit=True)
def ready_for_answer(message: types.Message, state_context: StateContext):
    with state_context.data() as data:
        msg = f"Your name is {data['name']}, surname is {data['surname']}, age is {message.text}"

    bot.send_message(message.chat.id, msg, parse_mode="html", reply_to_message_id=message.message_id)
    state_context.delete()


@bot.message_handler(state=MyStates.age, is_digit=False)
def age_incorrect(message: types.Message):
    bot.send_message(message.chat.id, 'Looks like you are submitting a string in the field age. Please enter a number', reply_to_message_id=message.message_id)


bot.add_custom_filter(custom_filters.StateFilter(bot))
bot.add_custom_filter(custom_filters.IsDigitFilter())

from telebot.states.sync.middleware import StateMiddleware

bot.setup_middleware(StateMiddleware(bot))

bot.infinity_polling(skip_pending=True)ending=True)