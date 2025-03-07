#date: 2025-03-07T16:50:06Z
#url: https://api.github.com/gists/16ea9d57fea1ac51e8602e238351f374
#owner: https://api.github.com/users/unbracketed

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "nanodjango",
#     "channels",
#     "daphne",
#     "htpy",
#     "markdown",
#     "markupsafe",
#     "llm"
# ]
# ///

import json
import uuid

from channels.generic.websocket import WebsocketConsumer
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from django.http import HttpResponse
from django.urls import path
from markupsafe import Markup
from markdown import markdown
from htpy import (
    body,
    button,
    div,
    form,
    h1,
    head,
    html,
    input,
    meta,
    script,
    link,
    title,
    main,
    style,
    fieldset,
    article,
)
from nanodjango import Django
import llm


#  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
#
#  ┌┬┐┌─┐┌┬┐┌─┐┬  ┌─┐┌┬┐┌─┐
#   │ ├┤ │││├─┘│  ├─┤ │ ├┤ 
#   ┴ └─┘┴ ┴┴  ┴─┘┴ ┴ ┴ └─┘
def html_template():
    return html[
        head[
            meta(charset="utf-8"),
            meta(name="viewport", content="width=device-width, initial-scale=1"),
            title["llm chat"],
            script(src="https://unpkg.com/htmx.org@2.0.4"),
            script(src="https://unpkg.com/htmx-ext-ws@2.0.2"),
            script(src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"),
            link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/default.min.css"),
            script(src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"),
            script(src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/go.min.js"),
            script(src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"),
            script(src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/javascript.min.js"),
            script(src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/bash.min.js"),
            script(src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/sql.min.js"),
            script(src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/yaml.min.js"),
            script(src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/yaml.min.js"),

            link(
                rel="stylesheet",
                href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css",
            ),
            style[
                Markup("""
                .message { padding: .5rem;  }
                .user-message {
                    border: 1px solid #999;
                    border-radius: 0.5rem;
                    margin-bottom: 0.5rem;
                }
                .response-message { 
                    font-weight: bold; 
                    background-color: #333;
                    border: 1px solid green;
                    border-radius: 0.5rem;
                    margin-bottom: 1rem;
                }
                .markdown-content {
                    display: none;
                }
                """)
            ],
            script[
                Markup("""
                // Create a MutationObserver to watch for content changes in hidden elements
                const observer = new MutationObserver((mutations) => {
                    mutations.forEach((mutation) => {
                        if (mutation.type === 'childList' && mutation.target.classList.contains('markdown-content')) {
                            // Get the visible container (sibling of the hidden content)
                            const visibleContainer = mutation.target.nextElementSibling;
                            if (visibleContainer) {
                                visibleContainer.innerHTML = marked.parse(mutation.target.textContent);
                                visibleContainer.querySelectorAll('pre code').forEach((block) => {
                                    hljs.highlightElement(block);
                                });
                            }
                        }
                    });
                });

                // Start observing the message list for changes
                document.addEventListener('DOMContentLoaded', () => {
                    const messageList = document.getElementById('message-list');
                    if (messageList) {
                        observer.observe(messageList, {
                            childList: true,
                            subtree: true,
                            characterData: true
                        });
                    }
                });
                """)
            ],
        ],
        body[
            main(class_="container")[
                article[
                    h1["♚ code king"],
                    div(hx_ext="ws", ws_connect="/ws/echo/")[
                        div("#message-list"),
                        form(ws_send=True)[
                            fieldset(role="group")[
                                input(
                                    name="message",
                                    type="text",
                                    placeholder="Type your message...",
                                    autocomplete="off",
                                ),
                                button(
                                    class_="primary outline",
                                    type="submit",
                                    onclick="setTimeout(() => this.closest('form').querySelector('input[name=message]').value = '', 0)",
                                )["↩"],
                            ]
                        ],
                    ],
                ],
            ],
        ],
    ]

#  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
#
#   ┌─┐┌─┐┌┬┐┌─┐┌─┐┌┐┌┌─┐┌┐┌┌┬┐┌─┐
#   │  │ ││││├─┘│ ││││├┤ │││ │ └─┐
#   └─┘└─┘┴ ┴┴  └─┘┘└┘└─┘┘└┘ ┴ └─┘


def response_message(message_text, id):
    return div("#message-list", hx_swap_oob=f"beforeend:{id} .markdown-content")[message_text]


def formatted_response_message(message_text, id):
    return div(id, hx_swap_oob="outerHTML")[
        div(data_theme="dark", class_="message response-message")[
            Markup(markdown(message_text, extensions=['fenced_code']))
        ]
    ]


def response_container(id, classname="response-message"):
    return div("#message-list", hx_swap_oob="beforeend")[
        div(id, class_=["message", classname], data_theme="dark")[
            div(class_="markdown-content")[""],  # Hidden element for raw markdown
            div(class_="rendered-content")[""]   # Visible element for rendered HTML
        ]
    ]


def user_message(message_text):
    return div("#message-list", hx_swap_oob="beforeend")[
        div(class_=["message", "user-message"])[
            message_text
        ]
    ]

#  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
#
#   ┬  ┬┬┌─┐┬ ┬┌─┐
#   └┐┌┘│├┤ │││└─┐
#    └┘ ┴└─┘└┴┘└─┘


def index(request):
    return HttpResponse(html_template())

#  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
#
#     ┬ ┬┌─┐┌┐ ┌─┐┌─┐┌─┐┬┌─┌─┐┌┬┐
#     │││├┤ ├┴┐└─┐│ ││  ├┴┐├┤  │ 
#     └┴┘└─┘└─┘└─┘└─┘└─┘┴ ┴└─┘ ┴ 


def is_code_fence(line):
    return line.strip().startswith("```")

def get_code_language(line):
    if is_code_fence(line):
        lang = line.strip()[3:].strip()
        return lang if lang else None
    return None

class EchoConsumer(WebsocketConsumer):
    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message_text = text_data_json.get("message", "")
        if not message_text.strip():
            return
        user_message_html = user_message(message_text)
        self.send(text_data=user_message_html)
        
        response = get_model().prompt(message_text)
        current_container_id = f"#response-message-{str(uuid.uuid4())}"
        current_container_html = response_container(current_container_id)
        self.send(text_data=current_container_html)
        
        buffer = ""
        in_code_block = False
        code_language = None
        
        for chunk in response:
            buffer += chunk
            lines = buffer.split('\n')
            buffer = lines[-1]  # Keep the last partial line in the buffer
            
            for line in lines[:-1]:  # Process all complete lines
                if is_code_fence(line):
                    if not in_code_block:
                        # Start of code block
                        in_code_block = True
                        code_language = get_code_language(line)
                        # Create new code block container
                        code_container_id = f"#code-block-{str(uuid.uuid4())}"
                        code_container_html = response_container(code_container_id, "code-block")
                        self.send(text_data=code_container_html)
                        # Send the fence line
                        self.send(text_data=response_message(line + '\n', code_container_id))
                    else:
                        # End of code block
                        in_code_block = False
                        code_language = None
                        # Send the closing fence
                        self.send(text_data=response_message(line + '\n', code_container_id))
                        # Create new response container for any following text
                        current_container_id = f"#response-message-{str(uuid.uuid4())}"
                        current_container_html = response_container(current_container_id)
                        self.send(text_data=current_container_html)
                else:
                    # Regular content
                    container_id = code_container_id if in_code_block else current_container_id
                    self.send(text_data=response_message(line + '\n', container_id))
        
        # Handle any remaining content in the buffer
        if buffer:
            container_id = code_container_id if in_code_block else current_container_id
            self.send(text_data=response_message(buffer, container_id))

#  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
#
#     ╔═╗╔═╗╔═╗
#     ╠═╣╠═╝╠═╝
#     ╩ ╩╩  ╩  
app = Django(
    # EXTRA_APPS=[
    #     "channels",
    # ],
    #
    # Nanodjango doesn't yet support prepending "priority" apps to INSTALLED_APPS,
    # and `daphne` must be the first app in INSTALLED_APPS.
    INSTALLED_APPS=[
        "daphne",
        "django.contrib.admin",
        "django.contrib.auth",
        "django.contrib.contenttypes",
        "django.contrib.sessions",
        "django.contrib.messages",
        "whitenoise.runserver_nostatic",
        "django.contrib.staticfiles",
        "channels",
    ],
    CHANNEL_LAYERS={
        "default": {
            "BACKEND": "channels.layers.InMemoryChannelLayer",
        },
    },
    ASGI_APPLICATION="__main__.htmx_websocket_interface",
)


#  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
#
#   ┬ ┬┬─┐┬  ┌─┐
#   │ │├┬┘│  └─┐
#   └─┘┴└─┴─┘└─┘
app.route("/")(index)
websocket_urlpatterns = [
    path("ws/echo/", EchoConsumer.as_asgi()),
]


htmx_websocket_interface = ProtocolTypeRouter(
    {
        "http": app.asgi,
        "websocket": AuthMiddlewareStack(URLRouter(websocket_urlpatterns)),
    }
)


#  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
#
#     ┬  ┬  ┌┬┐
#     │  │  │││
#     ┴─┘┴─┘┴ ┴
_model = None
def get_model():
    global _model
    if _model is None:
        model = llm.get_model()
        _model = model.conversation()
    return _model


# ^^^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^ ^^^^ 
#
if __name__ == "__main__":
    app.run()
