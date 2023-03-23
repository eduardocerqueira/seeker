#date: 2023-03-23T17:09:00Z
#url: https://api.github.com/gists/477ae2839400ef6f1a94027f8163455f
#owner: https://api.github.com/users/jonmatthis

# originally based on - https://docs.pinecone.io/docs/gen-qa-openai
import subprocess

from pinecone import Index

try:
    import os
    from time import sleep
    import openai
    from rich import print
    from dotenv import load_dotenv
    from datasets import load_dataset
    from tqdm.auto import tqdm
    import pinecone
except ImportError as e:
    print(e)
    print("Error importing dependencies. Please install them with the comand: \n ```\npip install openai, rich, python-dotenv, pinecone-client, datasets, tqdm\n```")
    exit(1)


#make a file named `.env` in the same directory as this file and add your API keys there so it looks like

"""
#.env file
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx #get this from https://platform.openai.com/account/api-keys
PINECONE_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx #get this from app.pinecone.io
PINECONE_ENVIRONMENT=us-west-2 #or whatever environment shows up on your Pinecone dashboard
"""

print("Loading environment variables")
load_dotenv()

# get API key from top-right dropdown on OpenAI website
openai.api_key = os.getenv("OPENAI_API_KEY")

# initialize connection to pinecone (get API key at app.pinecone.io)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")


def complete(prompt):
    # query text-davinci-003
    completion_response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens= "**********"
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return completion_response


def retrieve(query:str, index:Index, limit: int = 3750, embed_model: str = "text-embedding-ada-002"):
    print(
        f"Running - embedding_create_response = openai.Embedding.create(input=texts, engine=embed_model): \ntext={[query]}\nengine={embed_model}")
    embedding_create_response = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    embedding_vector_of_the_query = embedding_create_response['data'][0]['embedding']
    print(
        f"embedding_vector = embedding_create_response['data'][0]['embedding']: len(embedding_vector) = \n {len(embedding_vector_of_the_query)}")

    # get relevant contexts
    print(f"get relevant contexts (including the questions):")
    embedded_query_response = index.query(embedding_vector_of_the_query, top_k=3, include_metadata=True)
    contexts = [
        x['metadata']['text'] for x in embedded_query_response['matches']
    ]
    print(f"contexts = [x['metadata']['text'] for x in embedded_query_response['matches']]: \n {contexts}")

    # build our prompt with the retrieved contexts included
    prompt_start_string = "Answer the question based on the context below.\n\nContext:\n "

    prompt_end_string = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                    prompt_start_string +
                    "\n\n---\n\n".join(contexts[:i - 1]) +
                    prompt_end_string
            )
            break
        elif i == len(contexts) - 1:
            prompt = (
                    prompt_start_string +
                    "\n\n---\n\n".join(contexts) +
                    prompt_end_string
            )
    return prompt

def main():
    query = (
            "Which training method should I use for sentence transformers when " +
            "I only have pairs of related sentences?"
    )

    print(f"Query: {query}")
    response = complete(query)
    print(f"Response: {response}\n----\n")
    print(f"Response Message: {response['choices'][0]['text'].strip()}")

    embed_model = "text-embedding-ada-002"

    embedding_response = openai.Embedding.create(
        input=[
            "Sample document text goes here",
            "there will be several phrases in each batch"
        ], engine=embed_model
    )

    # print(f"\n----\nEmbedding Response:\n {embedding_response}\n----\n")



    data = load_dataset('jamescalam/youtube-transcriptions', split='train')

    print(f"Printing dataset: {data}")
    print(f"Printing dataset[0]: {data[0]}")

    print(
        "The dataset contains many small snippets of text data. We will need to merge many snippets from each video to create more substantial chunks of text that contain more information.")


    new_data = []

    window = 20  # number of sentences to combine
    stride = 4  # number of sentences to 'stride' over, used to create overlap

    for i in tqdm(range(0, len(data), stride)):
        i_end = min(len(data) - 1, i + window)
        if data[i]['title'] != data[i_end]['title']:
            # in this case we skip this entry as we have start/end of two videos
            continue
        text = ' '.join(data[i:i_end]['text'])
        # create the new merged dataset
        new_data.append({
            'start': data[i]['start'],
            'end': data[i_end]['end'],
            'title': data[i]['title'],
            'text': text,
            'id': data[i]['id'],
            'url': data[i]['url'],
            'published': data[i]['published'],
            'channel_id': data[i]['channel_id']
        })

    print(f"new_data[0]: {new_data[0]}")

    print("Indexing Data in Vector DB - Now we need a place to store these embeddings and enable a efficient vector "
          "search through them all. To do that we use Pinecone, we can get a free API key and enter it "
          "below where we will initialize our connection to Pinecone and create a new index. You can find "
          "your environment in the Pinecone console under API Keys.")


    index_name = 'openai-youtube-transcriptions'


    print("Initializing Pinecone client")
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    print("Pinecone client initialized!")
    print(f"pinecone.list_indexes(): {pinecone.list_indexes()}")
    # check if index already exists (it shouldn't if this is first time)
    if index_name not in pinecone.list_indexes():
        # if does not exist, create index
        print(f"Creating index: {index_name}...")
        pinecone.create_index(
            index_name,
            dimension=len(embedding_response['data'][0]['embedding']),
            metric='cosine',
            metadata_config={'indexed': ['channel_id', 'published']}
        )
        print(f"Index created: {index_name}")

    print(f"Connect to index: {index_name}")
    index = pinecone.Index(index_name)
    # view index stats
    print(f"index.describe_index_stats(): {index.describe_index_stats()}")

    if index.describe_index_stats()['total_vector_count'] == 0:
        print("We can see the index is currently empty with "
              "a total_vector_count of 0. We can begin "
              "populating it with OpenAI text-embedding-ada-002 built embeddings...")


        batch_size = 100  # how many embeddings we create and insert at once

        for i in tqdm(range(0, len(new_data), batch_size)):
            # find end of batch
            i_end = min(len(new_data), i + batch_size)
            meta_batch = new_data[i:i_end]
            # get ids
            ids_batch = [x['id'] for x in meta_batch]
            # get texts to encode
            texts = [x['text'] for x in meta_batch]
            # create embeddings (try-except added to avoid RateLimitError)
            try:
                res = openai.Embedding.create(input=texts, engine=embed_model)
            except:
                done = False
                while not done:
                    sleep(5)
                    try:
                        res = openai.Embedding.create(input=texts, engine=embed_model)
                        done = True
                    except:
                        pass
            embeds = [record['embedding'] for record in res['data']]
            # cleanup metadata
            meta_batch = [{
                'start': x['start'],
                'end': x['end'],
                'title': x['title'],
                'text': x['text'],
                'url': x['url'],
                'published': x['published'],
                'channel_id': x['channel_id']
            } for x in meta_batch]
            to_upsert = list(zip(ids_batch, embeds, meta_batch))
            # upsert to Pinecone
            index.upsert(vectors=to_upsert)




    print("first we retrieve relevant items from Pinecone...")
    prompt_with_context = retrieve(query, index)
    print(f"prompt_with_context: {prompt_with_context}")

    print("then we use the retrieved contexts to generate an answer...")
    response_from_prompt_with_context = complete(prompt_with_context)
    print(f"response_from_prompt_with_context: {response_from_prompt_with_context} \n \n --- \n \n")
    print(f"response_from_prompt_with_context['choices'][0]['text']: {response_from_prompt_with_context['choices'][0]['text']}")

    print(
        "\n\n And we get a pretty great answer straight away, specifying to use multiple-rankings loss (also called multiple negatives ranking loss)!")



if __name__ == '__main__':
    print(f"Running main() from {__file__}")
    main()