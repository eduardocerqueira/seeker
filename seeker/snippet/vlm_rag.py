#date: 2024-08-29T16:53:11Z
#url: https://api.github.com/gists/f4006d00cc1fcfa237d7f191c940011d
#owner: https://api.github.com/users/sovrasov

import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor
from PIL import Image
from io import BytesIO


if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.bfloat16
else:
    device = torch.device("cpu")
    dtype = torch.float32


from torch import nn
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration, PaliGemmaPreTrainedModel

class ColPali(PaliGemmaPreTrainedModel):
    def __init__(self, config):
        super(ColPali, self).__init__(config=config)
        self.model: PaliGemmaForConditionalGeneration = PaliGemmaForConditionalGeneration(config)
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.model.config.text_config.hidden_size, self.dim)
        self.main_input_name = "doc_input_ids"

    def forward(self, *args, **kwargs):
        """
        Forward pass through Llama and the linear layer for dimensionality reduction

        Args:
        - input_ids (torch.LongTensor): "**********"
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - torch.Tensor: "**********"
        """
        outputs = self.model(*args, output_hidden_states=True, **kwargs)
        last_hidden_states = outputs.hidden_states[-1]
        proj = self.custom_text_proj(last_hidden_states)
        # normalize l2 norm
        proj = proj / proj.norm(dim=-1, keepdim=True)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)
        return proj



model_name = "vidore/colpali"
model = ColPali.from_pretrained("google/paligemma-3b-mix-448", torch_dtype=dtype, device_map=device).eval()
model.load_adapter(model_name)
model.to(device)
processor = AutoProcessor.from_pretrained(model_name)

#BERT_Article.pdf: https://arxiv.org/pdf/1810.04805
#Transformers_Article.pdf: https://arxiv.org/pdf/1706.03762

pdfs = [{"file_name": "data/BERT_Article.pdf"}, {"file_name": "data/Transformers_Article.pdf"}]


import requests
from pdf2image import convert_from_path
from pypdf import PdfReader

def preprocessing(pdfs):
    documents = []
    images = []
    metadata = []
    for pdf in pdfs:
        file_name = pdf["file_name"]
        reader = PdfReader(file_name)
        for page_number in range(len(reader.pages)):
            page = reader.pages[page_number]
            text = page.extract_text()
            documents.append(text)
            metadata.append({"page": page_number, "file_path": file_name})
        images_for_file = convert_from_path(file_name)
        images += images_for_file
    assert len(images) == len(documents)
    assert len(metadata) == len(documents)
    return documents, images, metadata

documents, images, metadata = preprocessing(pdfs)

from pdf2image import convert_from_path
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

def indexing(images):
    ds = []
    dataloader = DataLoader(
        images,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: process_images(processor, x),
    )
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
    return ds

# Help function to process the images into the right (data) format
def process_images(processor, images, max_length: int = 50):
    texts_doc = ["Describe the image."] * len(images)
    images = [image.convert("RGB") for image in images]

    batch_doc = processor(
        text=texts_doc,
        images=images,
        return_tensors="pt",
        padding="longest",
        max_length=max_length + processor.image_seq_length,
    )
    return batch_doc

index = indexing(images)


# The model requires a mock image to be added to the query.
mock_image = Image.new("RGB", (448, 448), (255, 255, 255))

def search(query: str, index, documents, images, metadata, k=5):
    # text, images, and metadata are just passed without processing
    qs = []
    with torch.no_grad():
        batch_query = process_queries(processor, [query], mock_image)
        batch_query = {k: v.to(device) for k, v in batch_query.items()}
        embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))
    # run evaluation
    scores = evaluate_colbert(qs, index)
    relevant_pages = torch.topk(scores, k, dim=1, largest=True).indices
    relevant_pages = relevant_pages.squeeze()
    result = []
    for p in relevant_pages:
        result.append({"document": documents[p], "image": images[p], "score": scores[:,p].item(), "metadata": metadata[p]})
    return result

# Help function to process the queries into the right (data) format
def process_queries(processor, queries, mock_image, max_length: int = 50):
    texts_query = []
    for query in queries:
        query = f"Question: {query}<unused0><unused0><unused0><unused0><unused0>"
        texts_query.append(query)

    batch_query = processor(
        images=[mock_image.convert("RGB")] * len(texts_query),
        # NOTE: the image is not used in batch_query but it is required for calling the processor
        text=texts_query,
        return_tensors="pt",
        padding="longest",
        max_length=max_length + processor.image_seq_length,
    )
    del batch_query["pixel_values"]

    batch_query["input_ids"] = batch_query["input_ids"][..., processor.image_seq_length :]
    batch_query["attention_mask"] = batch_query["attention_mask"][..., processor.image_seq_length :]
    return batch_query

# Help function to calculate the scores between queries and documents
def evaluate_colbert(qs, ps, batch_size=128) -> torch.Tensor:
    scores = []
    for i in range(0, len(qs), batch_size):
        scores_batch = []
        qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i : i + batch_size], batch_first=True, padding_value=0).to(device)
        for j in range(0, len(ps), batch_size):
            ps_batch = torch.nn.utils.rnn.pad_sequence(
                ps[j : j + batch_size], batch_first=True, padding_value=0
            ).to(device)
            scores_batch.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2))
        scores_batch = torch.cat(scores_batch, dim=1).cpu()
        scores.append(scores_batch)
    scores = torch.cat(scores, dim=0)
    return scores

# Function for image processing
def scale_image(image: Image.Image, new_height: int = 1024) -> Image.Image:
    """
    Scale an image to a new height while maintaining the aspect ratio.
    """
    # Calculate the scaling factor
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)

    # Resize the image
    scaled_image = image.resize((new_width, new_height))

    return scaled_image

query = "How many transformers blocks in BERT Base? Justify your answer."
retrieved_documents = search(query=query, index=index, documents=documents, images=images, metadata=metadata, k=3)


from IPython.display import display, HTML
import io
import base64

def display_images(retrieved_documents):
    html = "<table><tr>"

    for r in retrieved_documents:
        img = r["image"]  # Assuming this is a PIL Image object
        title1 = f"File: {r['metadata']['file_path']}"  # Extracting the title from metadata
        title2 = f"Page: {r['metadata']['page']}"  # Extracting the title from metadata
        title3 = f"Score: {r['score']}"  # Extracting the title from metadata

        # Save the image to a BytesIO object
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')  # Save as PNG or any other format
        img_byte_arr.seek(0)  # Move to the beginning of the BytesIO object
        img_data = img_byte_arr.getvalue()
        img_base64 = base64.b64encode(img_data).decode('utf-8')  # Encode to base64

        # Create HTML for image with titles above
        html += f"""
        <td style='text-align: left;'>
            <div style='margin-bottom: 5px; font-size: 12px; color: black;'>
                {title1}<br>{title2}<br>{title3}
            </div>
            <img src='data:image/png;base64,{img_base64}' style='width: 200px;'>
        </td>
        """

    html += "</tr></table>"
    display(HTML(html))

# Example usage
#display_images(retrieved_documents)

import base64
import io

# Function to process images
def get_base64_image(img: str | Image.Image, add_url_prefix: bool = True) -> str:
    """
    Convert an image (from a filepath or a PIL.Image object) to a JPEG-base64 string.
    """
    if isinstance(img, str):
        img = Image.open(img)
    elif isinstance(img, Image.Image):
        pass
    else:
        raise ValueError("`img` must be a path to an image or a PIL Image object.")

    buffered = io.BytesIO()
    img.save(buffered, format="jpeg")
    b64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{b64_data}" if add_url_prefix else b64_data


# Format the images in the right format for the prompt
def convert_documents_to_prompt(retrieved_documents):
    images_for_vlm = []
    for r in retrieved_documents:
        images_for_vlm.append(
        {
            "type": "image_url",
            "image_url": {"url": get_base64_image(r["image"])}
        })
    return images_for_vlm

images_for_vlm = convert_documents_to_prompt(retrieved_documents)

images_raw = [r["image"] for r in retrieved_documents]


from openai import OpenAI

# Visual Language Model
def vlm(prompt, retrieved_documents):

    images_for_vlm = convert_documents_to_prompt(retrieved_documents)
    print(images_for_vlm)
    print(prompt)
    content = [{"type": "text", "text": prompt}] + images_for_vlm

    client = OpenAI()
    response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
            {
                "role": "user",
                "content": content
            }
        ],
      max_tokens= "**********"
    )
    return response.choices[0].message.content

from transformers import AutoProcessor, LlavaForConditionalGeneration

start = time.time()

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
prompt = f"USER: {'<image>'*len(images_raw)}\n{query} ASSISTANT:"
inputs = processor(text=prompt, images=images_raw, return_tensors="pt")
generate_ids = "**********"=100)
print(processor.batch_decode(generate_ids, skip_special_tokens= "**********"=False)[0])
print(f"Elapsed {time.time() - start}")


#from IPython.display import display, Markdown
#result = vlm(prompt=query, retrieved_documents=retrieved_documents)
#print(result)
#display(Markdown(result))uery, retrieved_documents=retrieved_documents)
#print(result)
#display(Markdown(result))