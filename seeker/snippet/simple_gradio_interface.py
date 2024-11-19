#date: 2024-11-19T16:46:22Z
#url: https://api.github.com/gists/c3140dec7302ec6cae3e0a1da1f176dc
#owner: https://api.github.com/users/agatheminaro

from __future__ import annotations

import gradio as gr
import numpy as np


def image_classifier(input_image: np.ndarray) -> dict[str, float]:
    """Output a dummy probabilitie for the image to be a dog or a cat."""
    return {"cat": 0.3, "dog": 0.7}


demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label")
demo.launch()