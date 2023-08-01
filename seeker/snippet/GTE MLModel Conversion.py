#date: 2023-08-01T16:48:53Z
#url: https://api.github.com/gists/1c1396192bb588801cd0a2f03358c1a7
#owner: https://api.github.com/users/ZachNagengast

import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
import coremltools as ct
import numpy as np

model_name = "thenlper/gte-large"

tokenizer = "**********"
model = AutoModel.from_pretrained(model_name)

import torch.nn.functional as F
from pprint import pprint

test_sentences = ["1 test sentence","2 test sentence","3 test sentence","test sentence 1 2 3 4",
                    "1 test sentence","2 test sentence","3 test sentence","test sentence 1 2 3 4"]

encoded_input = "**********"=512, padding="max_length", truncation=True, return_tensors='pt')

# Check model inputs
print(encoded_input.keys())
print(encoded_input['input_ids'].shape)

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input, return_dict=True)

# Check model outputs
print(model_output.keys())

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"f "**********"o "**********"r "**********"w "**********"a "**********"r "**********"d "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"i "**********"n "**********"p "**********"u "**********"t "**********"_ "**********"i "**********"d "**********"s "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"t "**********"y "**********"p "**********"e "**********"_ "**********"i "**********"d "**********"s "**********", "**********"  "**********"a "**********"t "**********"t "**********"e "**********"n "**********"t "**********"i "**********"o "**********"n "**********"_ "**********"m "**********"a "**********"s "**********"k "**********") "**********": "**********"
        with torch.no_grad():
            model_output = "**********"=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_dict=True)

        # Perform pooling for GTE models
        last_hidden = model_output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        return embeddings

# Initialize the wrapper with the original model
wrapped_model = ModelWrapper(model)

# Trace the model with both input_ids and attention_mask
traced_model = "**********"
traced_model.eval()

mlprogram = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, 512), dtype=np.int32),
        ct.TensorType(name= "**********"=(1, 512), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, 512), dtype=np.int32),
    ],
    outputs=[ct.TensorType(name="embeddings")],
    convert_to="mlprogram",
)

spec = mlprogram.get_spec()
outputmodel = ct.models.MLModel(spec, weights_dir=mlprogram.weights_dir)


saved_model = f'~/Downloads/{model_name}.mlpackage'
outputmodel.save(saved_model)

compressed_model = ct.compression_utils.affine_quantize_weights(outputmodel, mode="linear", dtype=ct.converters.mil.mil.types.int8)

compressed_model.save(f'~/Downloads/{model_name}-quantized-8-bit.mlpackage')