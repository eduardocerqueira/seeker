#date: 2022-08-01T17:11:53Z
#url: https://api.github.com/gists/b6285df658c24ec20c818817767809c4
#owner: https://api.github.com/users/pszemraj

"""
    test_viz.py - a basic script to test text2image with huggingface diffusers
"""
import argparse
import logging
import pprint as pp
import time
from pathlib import Path

from diffusers import DiffusionPipeline
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

def get_parser():
    """
    get_parser - a helper function for the argparse module
    """

    parser = argparse.ArgumentParser(
        description="basic test of the diffusion pipeline"
    )
    parser.add_argument(
        "-m",
        "--model",
        required=False,
        type=str,
        default="CompVis/ldm-text2im-large-256",
        help="model to use for text2im"
    )
    parser.add_argument(
        "-p",
        "--prompt",
        required=False,
        type=str,
        default="a nice zombie eating a steak burrito, as digital art",
        help="prompt to use for text2im",
    )
    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()
    logging.info(f"args: {pp.pformat(args)}")
    model_id = args.model
    prompt = args.prompt

    _here = Path(__file__).parent
    outdir = _here / "generated-images"
    outdir.mkdir(exist_ok=True)

    # load model and scheduler
    ldm = DiffusionPipeline.from_pretrained(model_id)
    logging.info("Loaded model: {}".format(model_id))
    # run pipeline in inference (sample random noise and denoise)
    st = time.perf_counter()
    images = ldm([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6)["sample"]
    rt = round(time.perf_counter() - st, 2)
    logging.info("Finished inference in {} seconds".format(rt))
    # save images
    for idx, image in enumerate(images):
        out_path = outdir / f"{prompt}_{idx}.png"
        image.save(out_path)
        image = Image.open(out_path)
        image.show()

    logging.info(f"Saved images to {outdir}")

