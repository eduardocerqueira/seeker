#date: 2024-11-19T17:04:12Z
#url: https://api.github.com/gists/ef4902a3d9b55a833354a3c84cab9828
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


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_name = gr.Textbox(label="Enter your name")
            input_temperature = gr.Slider(0, 100, label="What is the temperature?")
            input_is_morning = gr.Checkbox(label="Is it morning?")
        with gr.Column():
            greeting_output = gr.Textbox(label="Greeting")
            celsius_output = gr.Number(label="Temperature in Celsius")
    button = gr.Button(value="Greet")
    button.click(
        greet,
        inputs=[input_name, input_temperature, input_is_morning],
        outputs=[greeting_output, celsius_output],
    )

demo.launch()