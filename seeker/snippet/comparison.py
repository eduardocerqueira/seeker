#date: 2024-12-20T17:01:22Z
#url: https://api.github.com/gists/05481c1a52b74d3686e234420323b3ae
#owner: https://api.github.com/users/shawngraham

# and here we're going to load the models from huggingface and test them against the same output
# so you can see the difference between fine-tuning the original smol-135 makes
# this was trained on free-tier google colab for not very long on 800 rows of training data that I
# wrangled into correct shape. Training scripts etc will be shared in due course, but not bad for a first stab, eh?
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

def generate_response_pipeline(pipeline, input_text, max_length=500):
    """
    Generates a response using a Hugging Face pipeline
    """
    response = pipeline(input_text, max_length=max_length)[0]['generated_text']

    return response

 "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"n "**********"e "**********"r "**********"a "**********"t "**********"e "**********"_ "**********"r "**********"e "**********"s "**********"p "**********"o "**********"n "**********"s "**********"e "**********"( "**********"m "**********"o "**********"d "**********"e "**********"l "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********", "**********"  "**********"i "**********"n "**********"p "**********"u "**********"t "**********"_ "**********"t "**********"e "**********"x "**********"t "**********", "**********"  "**********"m "**********"a "**********"x "**********"_ "**********"l "**********"e "**********"n "**********"g "**********"t "**********"h "**********"= "**********"5 "**********"0 "**********"0 "**********") "**********": "**********"
    """
    Generates a response using the model
    """

    inputs = "**********"="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)

    response = "**********"=True)
    return response

def load_and_prepare_model(model_path, model_id=None, use_merged = True):
    """
    Loads and prepares either the merged model or the LoRA model
    """
    if use_merged:
         # Load the merged model directly
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
    else:
        # Load base model and LoRA weights
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
             device_map="auto",
            torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(model, model_path)

    tokenizer = "**********"
    tokenizer.pad_token = "**********"
    return model, tokenizer


def main():
    # Model names
    base_model_id = "HuggingFaceTB/SmolLM-135M"
    fine_tuned_model_id = "sgraham/merged_archae_metadata_model"  

     # Load the base model
    base_model, base_tokenizer = "**********"=base_model_id, model_id=base_model_id, use_merged = True)

   # Load the fine-tuned model
    ft_model, ft_tokenizer = "**********"=fine_tuned_model_id, model_id=base_model_id, use_merged = True)


    # Load the pipeline
    ft_pipeline = pipeline("text-generation", model=fine_tuned_model_id, device_map="auto", torch_dtype=torch.float16)


    # Test example (uses same format as used in fine-tuning)
    # location, companies, etc, all are all fake data here.
    prompt = """<|system|>You are a helpful archaeological assistant trained to categorize archaeological reports.\n<|user|>Please categorize this archaeological report metadata; return json: Report from Old Quarry Field at Emberton "This report documents archaeological findings from a survey led by Granite Digs at Hidden Valley located near Oakhaven. Key results include pottery analysis  details of a potential burial ground and a system of old roads." Ironmill Greenfield Emberton England 21483571 27700:390648 Subject:Archaeology Subject:Sherd Period:-800 - 1800 Subject:Excavations (Archaeology)--England Subject:Ditch Subject:Pit Subject:Strip Map And Sample Subject:Field Observation (Monitoring) Subject:Excavations (Archaeology)--England Period:ROMAN Associated ID: FAKE3 Import RCN: B45-546542 Creator:Starlight Research\n"""

    # Generate and compare
    print(f"Input Prompt: {prompt}")

    base_response = "**********"
    print(f"\nBase Model Response:\n{base_response}")

    ft_response_direct = "**********"
    print(f"\nFine-tuned Model Response (Direct):\n{ft_response_direct}")

    ft_response_pipeline = generate_response_pipeline(ft_pipeline, prompt)
    print(f"\nFine-tuned Model Response (Pipeline):\n{ft_response_pipeline}")


if __name__ == "__main__":
    main()