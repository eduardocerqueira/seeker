#date: 2023-09-11T17:00:26Z
#url: https://api.github.com/gists/79fb387d1fcb9b4c84c0f2f6c45e63c8
#owner: https://api.github.com/users/joshreini1

from llama_index import WikipediaReader

cities = [
    "Los Angeles", "Houston", "Honolulu", "Tucson", "Mexico City", 
    "Cincinatti", "Chicago"
]

wiki_docs = []
for city in cities:
    try:
        doc = WikipediaReader().load_data(pages=[city])
        wiki_docs.extend(doc)
    except Exception as e:
        print(f"Error loading page for city {city}: {e}")