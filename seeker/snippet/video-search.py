#date: 2024-06-07T16:49:02Z
#url: https://api.github.com/gists/5e30d6e210a74b2445ba26bf0cb48d4f
#owner: https://api.github.com/users/wrannaman

import subprocess
import json
import numpy as np
import tempfile
from PIL import Image as PILImage
from sentence_transformers import models, SentenceTransformer
import logging
import os
import io
from pymilvus import MilvusClient
from towhee import ops, pipe, DataCollection
from towhee.types.image import Image

# Map of IDs to categories
id_to_category = {
    '450140261493232760': 'kid',
    '450140261493232762': 'kid',
    '450140261493232764': 'kid',
    '450140261493232766': 'kid',
    '450140261493232768': 'kid',
    '450140261493232770': 'kid',
    '450140261493232772': 'kid',
    '450140261493232774': 'kid',
    '450140261493232776': 'kid',
    '450140261493232778': 'kid',
    '450140261493232780': 'pig',
    '450140261493232782': 'pig',
    '450140261493232784': 'pig',
    '450140261493232786': 'pig',
    '450140261493232788': 'pig',
    '450140261493232790': 'pig',
    '450140261493232792': 'pig',
    '450140261493232794': 'pig',
    '450140261493232796': 'pig',
    '450140261493232798': 'pig',
    '450140261493232800': 'flower',
    '450140261493232802': 'flower',
    '450140261493232804': 'flower',
    '450140261493232806': 'flower',
    '450140261493232808': 'flower',
    '450140261493232810': 'flower',
    '450140261493232812': 'flower',
    '450140261493232814': 'flower',
    '450140261493232816': 'flower',
    '450140261493232818': 'flower',
}


class console:
    @staticmethod
    def log(*args):
        print(*args)


def get_text_embedding(text):
    p2 = (
        pipe.input('text')
        .map('text', 'vec', ops.image_text_embedding.clip(model_name='clip_vit_base_patch16', modality='text'))
        .map('vec', 'vec', lambda x: x / np.linalg.norm(x))
        .output('text', 'vec')
    )
    col = DataCollection(p2(text))
    return col[0]['vec']


def pil_to_bytes(img):
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    with io.BytesIO() as output:
        img.save(output, format="JPEG")
        return output.getvalue()


def get_image_embedding(images):
    p1 = (
        pipe.input('img_bytes')
        # Decode bytes to image
        .map('img_bytes', 'img', ops.image_decode.cv2('rgb'))
        .map('img', 'vec', ops.image_text_embedding.clip(model_name='clip_vit_base_patch16', modality='image', device=0))
        .map('vec', 'vec', lambda x: x / np.linalg.norm(x))
        .output('vec')
    )
    results = []
    for img in images:
        img_bytes = pil_to_bytes(img)
        col = DataCollection(p1(img_bytes))
        results.append(col[0]['vec'])
    return results


def convert_to_pil_images(image_arrays):
    return [PILImage.fromarray(img) for img in image_arrays]


# Authentication enabled with a non-root user
client = MilvusClient(
    uri="https://in03-6bac0c4ac921d0f.api.gcp-us-west1.zillizcloud.com",
    # replace this with your token
    token="db_6bac0c4ac921d0f: "**********"
    db_name="videosearch"
)

has = client.has_collection(collection_name="test_vids")

if not has:
    client.create_collection(
        collection_name="test_vids",
        dimension=512,
        primary_field_name="id",
        id_type="string",
        vector_field_name="vector",
        metric_type="L2",
        auto_id=True,
        max_length=512
    )
else:
    console.log("Collection already exists")

text_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')


def pil_to_bytes(img):
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    with io.BytesIO() as output:
        img.save(output, format="JPEG")
        return output.getvalue()


class MediaEmbeddingExtractor:
    def get_scaled_size(self, width, height):
        target_width = 224
        w_percent = (target_width / float(width))
        h_size = int((float(height) * float(w_percent)))
        return target_width, h_size

    def probe(self, filename):
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries',
             'stream=width,height', '-of', 'json', filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return json.loads(result.stdout)

    def get_video_frames(self, content):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.flush()

            probe = self.probe(f.name)
            video_info = next(
                s for s in probe['streams'] if 'width' in s and 'height' in s)
            width, height = self.get_scaled_size(
                int(video_info['width']), int(video_info['height']))

            process = subprocess.run(
                ['ffmpeg', '-i', f.name, '-vf', f'scale={width}:{height}', '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', 'pipe:1'],  # noqa
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            frames = (
                np
                .frombuffer(process.stdout, np.uint8)
                .reshape([-1, height, width, 3])
            )

            indexes = np.random.randint(frames.shape[0], size=10)
            return [frame for frame in frames[indexes, :]]

    def get_image_frames(self, content):
        img = PILImage.open(io.BytesIO(content))
        img = img.resize((224, 224))
        return [np.array(img)]

    def get_frames(self, content, file_type):
        if file_type in ['mp4', 'avi', 'mov']:
            return self.get_video_frames(content)
        elif file_type in ['jpg', 'jpeg', 'png', 'webp']:
            return self.get_image_frames(content)
        else:
            raise ValueError("Unsupported file type")

    def get_embeddings(self, frames):
        pil_images = convert_to_pil_images(frames)
        vectors = get_image_embedding(pil_images)
        return vectors

    def log_embeddings(self, embeddings):
        console.log("embeddings: %s", len(embeddings), len(embeddings[0]))

    def index_embeddings(self, embeddings):
        for i, embedding in enumerate(embeddings):
            insert = client.insert(collection_name="test_vids", data={
                "vector": embedding
            })
            console.log("insert:", insert)


def process_videos_in_directory(directory):
    extractor = MediaEmbeddingExtractor()
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            file_path = os.path.join(directory, filename)
            console.log("file_path:", file_path)
            file_type = os.path.splitext(file_path)[1][1:].lower()

            with open(file_path, 'rb') as f:
                content = f.read()

            frames = extractor.get_frames(content, file_type)
            console.log("frames in %s: %d", filename, len(frames))
            embeddings = extractor.get_embeddings(frames)
            extractor.index_embeddings(embeddings)


def query_text(query_str):
    query_vector = get_text_embedding(query_str)
    console.log("query_vector:", len(query_vector))
    search_params = {
        "metric_type": "L2",
        "params": {}
    }
    res = client.search(
        collection_name="test_vids",
        data=[query_vector],
        limit=2,
        search_params=search_params
    )
    for item in enumerate(res[0]):
        console.log(
            "category:", id_to_category[item[1]['id']], 'should be', query_str)
    return "ok"


def query_image(image_path):
    with open(image_path, 'rb') as f:
        content = f.read()
    img = PILImage.open(io.BytesIO(content))
    embeddings = get_image_embedding([img])
    query_vector = embeddings[0]
    console.log("query_vector:", len(query_vector))
    search_params = {
        "metric_type": "L2",
        "params": {}
    }
    res = client.search(
        collection_name="test_vids",
        data=[query_vector],
        limit=2,
        search_params=search_params
    )
    for item in enumerate(res[0]):
        console.log("category:", id_to_category[item[1]['id']], 'distance:', item[1]['distance'])  # noqa
    return res


# Usage example
if __name__ == "__main__":
    # index
    # process_videos_in_directory('../test_files')
    # console.log("query")
    # results = query_text("cake")
    # console.log("query text results", results)
    results = query_image("../test_files/flower-test.png")
    console.log("query image results", results)
lts)
