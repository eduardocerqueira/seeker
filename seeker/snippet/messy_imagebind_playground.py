#date: 2023-08-24T16:34:40Z
#url: https://api.github.com/gists/6199334d4e69902e52d27c10428d6f58
#owner: https://api.github.com/users/louis030195

import data
import torch
from models import imagebind_model
from models.imagebind_model import ModalityType
import uvicorn
from fastapi import FastAPI
import requests


device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)



app = FastAPI()


@app.post("/")
def predict(req: dict):
    # body is like {
    #   text: ["A dog.", "A car", "A bird"],
    #   image: ["https://images.com/foo.png", "https://images.com/bar.png", "https://images.com/baz.png"],
    #   audio: ["https://audios.com/foo.wav", "https://audios.com/bar.wav", "https://audios.com/baz.wav"],
    # }

    text_list = req.get("text", [])
    image_paths = req.get("image", [])
    audio_paths = req.get("audio", [])

    print("text_list:", text_list)
    print("image_paths:", image_paths)
    print("audio_paths:", audio_paths)

    # download image and audio
    # image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
    # audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav"]
    for path in image_paths + audio_paths:
        if path.startswith("http"):
            r = requests.get(path, timeout=10)
            with open("downloads/" + path.split("/")[-1], "wb") as f:
                f.write(r.content)
    
    # replace path with local path
    image_paths = ["downloads/" + path.split("/")[-1] for path in image_paths]
    audio_paths = ["downloads/" + path.split("/")[-1] for path in audio_paths]

    # Load data
    inputs = {}
    if text_list:
        inputs[ModalityType.TEXT] = data.load_and_transform_text(text_list, device)
    if image_paths:
        inputs[ModalityType.VISION] = data.load_and_transform_vision_data(image_paths, device)
    if audio_paths:
        inputs[ModalityType.AUDIO] = data.load_and_transform_audio_data(audio_paths, device)

    with torch.no_grad():
        embeddings = model(inputs)

    vision_x_text = torch.softmax(
        embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1
    ) if ModalityType.VISION in embeddings and ModalityType.TEXT in embeddings else None
    print(
        "Vision x Text: ",
        vision_x_text,
    )
    audio_x_text = torch.softmax(
        embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1
    ) if ModalityType.AUDIO in embeddings and ModalityType.TEXT in embeddings else None
    print(
        "Audio x Text: ",
        audio_x_text,
    )
    vision_x_audio = torch.softmax(
        embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1
    ) if ModalityType.VISION in embeddings and ModalityType.AUDIO in embeddings else None
    print(
        "Vision x Audio: ",
        vision_x_audio,
    )
    # return {
    #     "vision_x_text": vision_x_text.tolist(),
    #     "audio_x_text": audio_x_text.tolist(),
    #     "vision_x_audio": vision_x_audio.tolist(),
    # }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
