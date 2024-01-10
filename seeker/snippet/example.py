#date: 2024-01-10T17:00:53Z
#url: https://api.github.com/gists/90f6c13688e68bb9cb83e9cbfd0312d3
#owner: https://api.github.com/users/andrewfrench

import os

import boto3

from griptape.drivers import (
    AmazonBedrockImageGenerationDriver,
    BedrockStableDiffusionImageGenerationModelDriver,
    OpenAiDalleImageGenerationDriver
)
from griptape.tasks import VariationImageGenerationTask, PromptImageGenerationTask, BaseTask
from griptape.engines import VariationImageGenerationEngine, PromptImageGenerationEngine
from griptape.structures import Pipeline
from griptape.artifacts import ImageArtifact, TextArtifact
from griptape.loaders.image_loader import ImageLoader


os.environ["OPENAI_API_KEY"] = "your-key"
boto3_session = boto3.Session() # your AWS profile info here

prompt = TextArtifact("a golden doodle puppy")

generation_task_output = "/Users/andrew/downloads/base-generation.png"
pixel_variation_task_output = "/Users/andrew/downloads/pixel-variation.png"
anime_variation_task_output = "/Users/andrew/downloads/anime-variation.png"

generation_task = PromptImageGenerationTask(
    input=prompt,
    image_generation_engine=PromptImageGenerationEngine(
        image_generation_driver=OpenAiDalleImageGenerationDriver(
            model="dall-e-2",
            image_size="512x512",
        ),
    ),
    output_file=generation_task_output,
)


def returns_variation_task_input(task: BaseTask) -> (str, ImageArtifact):
    return prompt, ImageLoader().load(generation_task_output)


pixel_variation_task = VariationImageGenerationTask(
    input=returns_variation_task_input,
    image_generation_engine=VariationImageGenerationEngine(
        image_generation_driver=AmazonBedrockImageGenerationDriver(
            image_generation_model_driver=BedrockStableDiffusionImageGenerationModelDriver(
                style_preset="pixel-art",
            ),
            model="stability.stable-diffusion-xl-v0",
            session=boto3_session,
            image_width=512,
            image_height=512,
        ),
    ),
    output_file=pixel_variation_task_output,
)

anime_variation_task = VariationImageGenerationTask(
    input=returns_variation_task_input,
    image_generation_engine=VariationImageGenerationEngine(
        image_generation_driver=AmazonBedrockImageGenerationDriver(
            image_generation_model_driver=BedrockStableDiffusionImageGenerationModelDriver(
                style_preset="anime",
            ),
            model="stability.stable-diffusion-xl-v0",
            session=boto3_session,
            image_width=512,
            image_height=512,
        ),
    ),
    output_file=anime_variation_task_output,
)

Pipeline(tasks=[generation_task, pixel_variation_task, anime_variation_task]).run()
