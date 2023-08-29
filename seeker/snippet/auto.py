#date: 2023-08-29T16:55:32Z
#url: https://api.github.com/gists/08fb103fc3e85bcb53fb7882837066f3
#owner: https://api.github.com/users/ddrscott

"""
Requirements:
pip install click langchain openai
"""
import sys
import click
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
class CustomHandler(StreamingStdOutCallbackHandler):
    def on_llm_start(self, serialized, prompts, **_) -> None:
        pass

    def on_llm_new_token(self, token: "**********":
        click.echo(token, nl= "**********"

    def on_llm_end(self, response, **kwargs) -> None:
        click.echo('\n')

from langchain.chat_models import ChatOpenAI

def auto_lint(data, model):
    llm=ChatOpenAI(
            client=None,
            model=model,
            temperature=0.1,
            verbose=True,
            callbacks=[CustomHandler()],
            streaming=True,
        )

    llm.predict(f"""You are an expert Python developer.
Please make the following updates to the attached code:
- add useful Google style docstrings to functions.
- fix spelling mistakes.
- strip whitespace.

```python
{data}
```
Updated Python code:""")

@click.command()
@click.option('--model', '-m', default='gpt-3.5-turbo-16k-0613')
@click.argument('src',
              type=click.File('r'),
              default=sys.stdin)
def my_command(model, src):
    data = None
    with src:
        data = src.read()
        auto_lint(data, model)

if __name__ == '__main__':
    my_command()
