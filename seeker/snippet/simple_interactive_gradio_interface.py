#date: 2024-11-19T16:57:06Z
#url: https://api.github.com/gists/e0d4c68957adbdb0d8a80117279ef6c9
#owner: https://api.github.com/users/agatheminaro

import gradio as gr


def greet(
    name: str,
    temperature: int,
    is_morning: bool,
) -> tuple[str, float]:
    """Greet the user with a message and the temperature in Celsius."""
    salutation = "Good morning" if is_morning else "Good evening"
    greeting = f"{salutation} {name}. It is {temperature} Fahrenheit today."
    celsius = round((temperature - 32) * 5 / 9, 2)
    return greeting, celsius


demo = gr.Interface(
    fn=greet,
    inputs=[
        gr.Text(label="What is your name?"),
        gr.Slider(0, 100, label="What is the temperature?"),
        gr.Checkbox(label="Is it morning?"),
    ],
    outputs=[gr.Text(label="Greeting"), gr.Number(label="Temperature in Celsius")],
)
if __name__ == "__main__":
    demo.launch()