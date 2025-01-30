#date: 2025-01-30T16:40:44Z
#url: https://api.github.com/gists/6981d896c87f391cebe20bfcc8bdaa97
#owner: https://api.github.com/users/davidberenstein1957


# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "ai-gradio[together]"
# ]
# ///
import gradio as gr
import ai_gradio

gr.load(
  name='together:deepseek-ai/DeepSeek-R1',
  src=ai_gradio.registry,
).launch()
