#date: 2024-10-28T16:57:02Z
#url: https://api.github.com/gists/3d73b8762c7c843b32c898f14c559259
#owner: https://api.github.com/users/markpbaggett

import httpx
import asyncio
from tqdm import tqdm

solr_instance = 'https://api.library.tamu.edu/solr/sage-core/select?indent=on&q=collection:"https://api.library.tamu.edu/fcrepo/rest/3b/6f/c3/25/3b6fc325-f6ca-41d8-b91e-8c5db3be8c13/apfelbaum-ww1_objects"&wt=json&fl=manifest,id&rows=10000'


async def fetch_manifest(client, url):
    try:
        response = await client.get(url, timeout=114)
        return response.json()
    except httpx.RequestError as e:
        print(f"An error occurred while requesting {url}: {e}")
        return None


async def main():
    async with httpx.AsyncClient(timeout=httpx.Timeout(114)) as client:
        # Fetch main data
        response = await client.get(solr_instance)
        data = response.json()

        # Create a list of background tasks for each document
        tasks = [asyncio.create_task(fetch_manifest(client, document['manifest'])) for document in data['response']['docs']]

        # Use tqdm to visualize progress without awaiting each task individually
        manifests = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await task
            if result is not None:
                manifests.append(result)


# Run the async main function
asyncio.run(main())
