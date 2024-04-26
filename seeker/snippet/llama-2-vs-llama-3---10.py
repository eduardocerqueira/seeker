#date: 2024-04-26T16:50:20Z
#url: https://api.github.com/gists/dd522ae82d669cab843f9ef45b8668d8
#owner: https://api.github.com/users/BobMerkus

from langchain_core.language_models import BaseChatModel
from IPython.display import display, Markdown

def stream_response(messages: list[dict], model: BaseChatModel, add_message: bool = True):
    """Stream the response of the model using a live markdown display and add the response to the messages."""
    model_name = f"### {len(messages)-1} - Running model `{model.name}`\n"
    result = ''
    display_handle = display(Markdown(""), display_id=True)
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"i "**********"n "**********"  "**********"m "**********"o "**********"d "**********"e "**********"l "**********". "**********"s "**********"t "**********"r "**********"e "**********"a "**********"m "**********"( "**********"m "**********"e "**********"s "**********"s "**********"a "**********"g "**********"e "**********"s "**********") "**********": "**********"
        result += "**********"
        display_handle.update(Markdown(model_name + result))
    if add_message:
        messages.append({"role": "ai", "content": result})
    return result

result = stream_response(messages, llama_2, add_message=False)