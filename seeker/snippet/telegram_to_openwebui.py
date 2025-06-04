#date: 2025-06-04T16:55:57Z
#url: https://api.github.com/gists/1aa9903a315c7957e63c1c7676bbf2c4
#owner: https://api.github.com/users/lemassykoi

import asyncio
import logging
import os
import json
import aiohttp
import re
import html

from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.enums import ChatAction, ParseMode
from aiogram.client.default import DefaultBotProperties

# --- Configuration ---
TELEGRAM_BOT_TOKEN = "**********"
OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY", "YOUR_OPENWEBUI_API_KEY_PLACEHOLDER")
OPENWEBUI_BASE_URL = "http://127.0.0.1:8080"
CHAT_COMPLETIONS_ENDPOINT = "/api/chat/completions"
MODELS_ENDPOINT = "/api/models"

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FSM States ---
class UserStates(StatesGroup):
    choosing_model = State()
    chatting = State()

# --- Bot Initialization ---
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# --- Helper Functions ---
async def fetch_openwebui_models(session: aiohttp.ClientSession):
    """Fetches available models from Open-WebUI."""
    headers = {
        "Authorization": f"Bearer {OPENWEBUI_API_KEY}"
    }
    url = f"{OPENWEBUI_BASE_URL}{MODELS_ENDPOINT}"
    try:
        logger.info(f"Attempting to fetch models from: {url}")
        async with session.get(url, headers=headers) as response:
            logger.info(f"OpenWebUI Models API response status: {response.status}")
            response.raise_for_status() # Check for HTTP errors first
            
            raw_response_text = await response.text()
            # Limit logging of potentially very long responses
            log_text = raw_response_text[:1000] + ('...' if len(raw_response_text) > 1000 else '')
            logger.debug(f"OpenWebUI Models API raw response text (up to 1000 chars): {log_text}") # INFO to DEBUG

            try:
                data = json.loads(raw_response_text)
            except json.JSONDecodeError as e_json:
                logger.error(f"JSONDecodeError for OpenWebUI Models API response: {e_json}")
                logger.error(f"Problematic text (up to 1000 chars): {log_text}")
                return []

            logger.debug(f"Parsed data type from Models API: {type(data)}") # INFO to DEBUG
            if isinstance(data, dict):
                logger.debug(f"Parsed data (dict) keys: {list(data.keys())}") # INFO to DEBUG
            elif isinstance(data, list):
                logger.debug(f"Parsed data is a list of length: {len(data)}") # INFO to DEBUG
            else:
                logger.debug(f"Parsed data is neither dict nor list: {str(data)[:500]}") # INFO to DEBUG

            # Path A: Check if data is a dict and contains a 'data' key which is a list
            if isinstance(data, dict) and 'data' in data and isinstance(data.get('data'), list):
                logger.debug("Parsing models using Path A (revised): data['data'] is a list.") # INFO to DEBUG
                return [model.get('id', model.get('name')) for model in data['data'] if model.get('id') or model.get('name')]
            # Path B: Check if data itself is a list (original Path B)
            elif isinstance(data, list): 
                logger.debug("Parsing models using Path B: data is a list.") # INFO to DEBUG
                return [model.get('id', model.get('name')) for model in data if model.get('id') or model.get('name')]
            # Path D (was Path A): Check if data is a dict and contains a 'models' key which is a list (original Path A)
            elif isinstance(data, dict) and 'models' in data and isinstance(data.get('models'), list):
                logger.debug("Parsing models using Path D (original Path A): data['models'] is a list.") # INFO to DEBUG
                return [model.get('id', model.get('name')) for model in data['models'] if model.get('id') or model.get('name')]
            
            logger.warning(f"Unexpected model list format (Path C - fallback). Data (up to 500 chars): {str(data)[:500]}")
            return []
    except aiohttp.ClientResponseError as e_http: 
        logger.error(f"HTTP error fetching models from Open-WebUI: Status={e_http.status}, Message='{e_http.message}'")
        try:
            error_body = await e_http.text(errors='ignore') # type: ignore
            logger.error(f"HTTP error body (up to 500 chars): {error_body[:500]}")
        except Exception as e_text:
            logger.error(f"Could not get error body: {e_text}")
        return []
    except aiohttp.ClientError as e_client: 
        logger.error(f"ClientError (e.g., network issue) fetching models from Open-WebUI: {e_client}")
        return []
    except Exception as e_general: # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred in fetch_openwebui_models: {e_general}", exc_info=True)
        return []

async def get_openwebui_chat_response(session: aiohttp.ClientSession, model_id: str, user_message: str):
    """Gets a chat response from Open-WebUI."""
    headers = {
        "Authorization": f"Bearer {OPENWEBUI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": user_message}],
        "stream": False
    }
    url = f"{OPENWEBUI_BASE_URL}{CHAT_COMPLETIONS_ENDPOINT}"
    try:
        async with session.post(url, headers=headers, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            # Assuming OpenAI compatible response structure
            if data.get("choices") and len(data["choices"]) > 0:
                message_content = data["choices"][0].get("message", {}).get("content")
                return message_content # This can be a string or None
            logger.warning(f"Unexpected chat response format: {data}")
            return "Sorry, I couldn't process that response." # Return a string for consistency
    except aiohttp.ClientError as e:
        logger.error(f"Error getting chat response from Open-WebUI: {e}")
        return f"Sorry, an API error occurred: {e}"
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from Open-WebUI chat endpoint: {e}")
        return "Sorry, I received an invalid response from the AI."
    except Exception as e_general: # Catch any other unexpected errors
        logger.error(f"An unexpected error in get_openwebui_chat_response: {e_general}", exc_info=True)
        return "An unexpected error occurred while getting the AI response."


# --- Handlers ---
@dp.message(CommandStart())
async def cmd_start(message: types.Message, state: FSMContext, session: aiohttp.ClientSession): # bot parameter removed, not used
    await state.clear() # Clear any previous state
    if message.from_user:
        logger.info(f"User {message.from_user.id} started interaction.")
    else:
        logger.info("Started interaction with a user with no ID.")
    await message.answer("Welcome! Let's choose an Ollama model to chat with.")

    fetched_models = await fetch_openwebui_models(session)
    if not fetched_models:
        await message.answer("Sorry, I couldn't fetch the list of available models right now. Please try again later or check my logs.")
        return

    # Filter models
    keywords_to_filter = ["embed", "impact", "code", "gemini_"]
    filtered_models = []
    for model_name in fetched_models:
        if model_name: # Ensure model_name is not None
            lowercase_model_name = model_name.lower()
            if not any(keyword in lowercase_model_name for keyword in keywords_to_filter):
                filtered_models.append(model_name)
    
    if not filtered_models:
        await message.answer("No suitable models available after filtering. Please check Open-WebUI or adjust filter keywords.")
        return

    # Sort models alphabetically
    sorted_models = sorted(filtered_models)
    
    # Store the filtered and sorted models in FSM context
    # This is what will be used for button generation and callback lookup
    models = sorted_models # Use 'models' as the variable name for consistency with below

    # Store the fetched models in FSM context to retrieve by index later
    await state.update_data(available_models=models)

    builder = InlineKeyboardBuilder()
    for index, model_id_text in enumerate(models):
        if model_id_text: # Ensure model_id_text is not None or empty
            # Use index in callback_data to keep it simple and avoid invalid characters
            builder.button(text=model_id_text, callback_data=f"modelidx_{index}")
    builder.adjust(2) # Adjust to show 2 buttons per row

    if not builder.buttons:
        await message.answer("No models seem to be available or the format was unexpected. Please check Open-WebUI.")
        return

    await message.answer("Please select a model:", reply_markup=builder.as_markup())
    await state.set_state(UserStates.choosing_model)

@dp.callback_query(StateFilter(UserStates.choosing_model), lambda c: c.data is not None and c.data.startswith("modelidx_")) # Changed prefix
async def process_model_callback(callback_query: types.CallbackQuery, state: FSMContext, bot: Bot): # Added bot
    if not callback_query.data: # Should be caught by lambda, but defensive
        await callback_query.answer("Error: No callback data received.", show_alert=True)
        return
    
    try:
        index_str = callback_query.data.split("modelidx_", 1)[1]
        model_index = int(index_str)

        user_data = await state.get_data()
        available_models = user_data.get("available_models")

        if not available_models or model_index >= len(available_models):
            logger.error(f"Invalid model index {model_index} or missing available_models in FSM.")
            error_message_text = "Error: Could not retrieve selected model. Please try /start again."
            if callback_query.message and isinstance(callback_query.message, types.Message):
                try:
                    await callback_query.message.edit_text(error_message_text)
                except Exception as edit_e:
                    logger.error(f"Error editing message (invalid index path): {edit_e}")
                    await bot.send_message(callback_query.from_user.id, error_message_text) # Fallback
            else:
                await bot.send_message(callback_query.from_user.id, error_message_text) # Fallback
            await callback_query.answer("Error processing selection.", show_alert=True)
            await state.clear() # Clear state as it's inconsistent
            return

        model_id = available_models[model_index]
    except (ValueError, IndexError) as e:
        logger.error(f"Error processing model selection callback: {e}. Data: {callback_query.data}")
        error_message_text = "Error: Invalid selection. Please try /start again."
        if callback_query.message and isinstance(callback_query.message, types.Message):
            try:
                await callback_query.message.edit_text(error_message_text)
            except Exception as edit_e:
                logger.error(f"Error editing message (value/index error path): {edit_e}")
                await bot.send_message(callback_query.from_user.id, error_message_text) # Fallback
        else:
            await bot.send_message(callback_query.from_user.id, error_message_text) # Fallback
        await callback_query.answer("Error processing selection.", show_alert=True)
        await state.clear() # Clear state
        return
        
    await state.update_data(selected_model=model_id)
    # await state.update_data(available_models=None) # Comment removed
    await state.set_state(UserStates.chatting)

    if callback_query.message and isinstance(callback_query.message, types.Message):
        try:
            await callback_query.message.edit_text(f"You've selected model: <b>{html.escape(model_id)}</b>. You can now start chatting!", parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"Error editing message: {e}")
            # Fallback if edit fails (e.g. message too old)
            await bot.send_message(callback_query.from_user.id, f"You've selected model: <b>{html.escape(model_id)}</b>. You can now start chatting!", parse_mode=ParseMode.HTML)

    await callback_query.answer() # Acknowledge the callback
    if callback_query.from_user:
        logger.info(f"User {callback_query.from_user.id} selected model: {model_id}")
    else:
        logger.error(f"User with no ID selected model: {model_id}")


@dp.message(StateFilter(UserStates.chatting))
async def handle_message(message: types.Message, state: FSMContext, session: aiohttp.ClientSession, bot: Bot): # Added bot
    user_data = await state.get_data()
    selected_model = user_data.get("selected_model")

    if not selected_model:
        await message.answer("Please select a model first using the /start command.")
        return

    if not message.text:
        await message.answer("Please send a text message.")
        return

    user_id_log = message.from_user.id if message.from_user else "Unknown"
    logger.info(f"User {user_id_log} (model: {selected_model}): {message.text}")
    
    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    raw_ai_response = await get_openwebui_chat_response(session, selected_model, message.text)
    # User's existing log line, changed to DEBUG as requested.
    logger.debug(f"Raw AI Response from get_openwebui_chat_response: {raw_ai_response}") # WARNING to DEBUG

    if raw_ai_response and isinstance(raw_ai_response, str):
        think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        match = think_pattern.search(raw_ai_response)

        thinking_process_str = None
        final_answer = raw_ai_response # Default to full response if no tags

        if match:
            thinking_process_str = match.group(1).strip()
            # Remove the matched <think>...</think> block and any surrounding newlines
            final_answer = think_pattern.sub("", raw_ai_response).strip() 
            
            if thinking_process_str: # Only send if there's content
                escaped_thinking = html.escape(thinking_process_str)
                #formatted_thinking = f"<pre>{escaped_thinking}</pre>"
                #formatted_thinking = f"<b>Thinking Process:</b>\n<span class='tg-spoiler'>{escaped_thinking}</span>"
                formatted_thinking = f"Thinking process:\n<blockquote expandable>{escaped_thinking}</blockquote>"
                try:
                    await message.answer(formatted_thinking, parse_mode=ParseMode.HTML)
                    logger.info(f"Sent thinking process to user {user_id_log}")
                except Exception as e_think:
                    logger.error(f"Error sending thinking process to {user_id_log}: {e_think}")
        
        if final_answer: # Only send if there's content
            try:
                # Uses default ParseMode.HTML set in Bot DefaultBotProperties
                await message.answer(final_answer) 
                logger.info(f"Sent final answer to user {user_id_log}")
            except Exception as e_ans:
                logger.error(f"Error sending final answer to {user_id_log}: {e_ans}")
                # Fallback message if sending final_answer fails
                await message.answer("Sorry, there was an issue formatting the AI's main response.")
        # If both thinking_process_str and final_answer are empty (e.g. response was ONLY <think></think>)
        elif not thinking_process_str and not final_answer: 
            logger.info(f"Both thinking process and final answer are empty for user {user_id_log}.")
            await message.answer("The AI's response was empty after processing. Please try again.")

    elif raw_ai_response: # Not None/empty, but not a string (should be handled by get_openwebui_chat_response)
         logger.warning(f"AI response was not a string for user {user_id_log}: {type(raw_ai_response)}")
         await message.answer("Received an unexpected response type from the AI.")
    else: # raw_ai_response is None or empty string
        logger.info(f"Received no response or empty one for user {user_id_log}.")
        await message.answer("I received no response or an empty one from the AI. Please try again.")

    log_snippet = (raw_ai_response[:100] + '...' if raw_ai_response and len(raw_ai_response) > 100 else raw_ai_response) if raw_ai_response else "N/A"
    logger.info(f"Bot interaction summary for user {user_id_log}: Raw AI response snippet: {log_snippet}")


@dp.message(StateFilter(None)) # Handles messages when no state is set (e.g., before /start)
async def handle_message_no_state(message: types.Message, state: FSMContext):
    await message.answer("Please use the /start command to begin and select a model.")


# --- Main Execution ---
async def main():
    bot_instance = Bot(
        token= "**********"
        default = DefaultBotProperties(
            parse_mode = ParseMode.HTML,
            link_preview_is_disabled = True
        ))

    async with aiohttp.ClientSession() as session:
        # Add the session to the dispatcher's shared context.
        # Handlers can then access it by declaring a parameter with the same name and type.
        dp['session'] = session
        # The bot_instance is passed positionally. Aiogram will inject it into handlers
        # that declare a 'bot: Bot' parameter.
        await dp.start_polling(bot_instance)


if __name__ == '__main__':
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"T "**********"E "**********"L "**********"E "**********"G "**********"R "**********"A "**********"M "**********"_ "**********"B "**********"O "**********"T "**********"_ "**********"T "**********"O "**********"K "**********"E "**********"N "**********"  "**********"= "**********"= "**********"  "**********"" "**********"Y "**********"O "**********"U "**********"R "**********"_ "**********"T "**********"E "**********"L "**********"E "**********"G "**********"R "**********"A "**********"M "**********"_ "**********"B "**********"O "**********"T "**********"_ "**********"T "**********"O "**********"K "**********"E "**********"N "**********"_ "**********"P "**********"L "**********"A "**********"C "**********"E "**********"H "**********"O "**********"L "**********"D "**********"E "**********"R "**********"" "**********"  "**********"o "**********"r "**********"  "**********"\ "**********"
       OPENWEBUI_API_KEY == "YOUR_OPENWEBUI_API_KEY_PLACEHOLDER":
        logger.warning("One or more API keys are placeholders. Please set them as environment variables or directly in the script.")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped manually.")
    except Exception as e:
        logger.critical(f"Bot failed to start or run: {e}", exc_info=True)
