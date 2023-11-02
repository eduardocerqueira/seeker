#date: 2023-11-02T16:59:58Z
#url: https://api.github.com/gists/d0069ccaffbb29747939ff6ddfaa6bca
#owner: https://api.github.com/users/Sumanth077

!pip install clarifai

# Set clarifai PAT as environment variable.
import os
os.environ["CLARIFAI_PAT"] = "<YOUR CLARIFAI PAT>"

from clarifai.client.app import App

# List all models in community filtered by model_type, description
all_llm_community_models = App().list_models(filter_by={"query": "LLM",
                                                        "model_type_id": "text-to-text"}, only_in_app=False)
all_llm_community_models = list(all_llm_community_models)