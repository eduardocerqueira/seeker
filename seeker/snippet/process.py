#date: 2025-02-28T16:56:17Z
#url: https://api.github.com/gists/8252a2f873f1bbd037301ee4ccae5de5
#owner: https://api.github.com/users/mekarpeles

import json

def flatten_books(data):
    flat_list = []
    
    def traverse(node, topics):
        if "sections" in node:
            for section in node["sections"]:
                new_topics = topics + [section["section"]]
                
                # Process entries at the current section level
                for entry in section.get("entries", []):
                    if entry.get("url"):
                        flat_list.append({
                            "title": entry["title"],
                            "author": entry.get("author"),
                            "language": node["language"].get("name") if node.get('language') else "",
                            "url": entry["url"],
                            "topics": new_topics,
                            "notes": entry.get("notes", [])
                        })
                
                # Process subsections recursively
                for subsection in section.get("subsections", []):
                    traverse(subsection, new_topics)
    
    # Start traversing from the root
    for child in data.get("children", []):
        if child["type"] == "books":
            for book in child.get("children", []):
                traverse(book, [])

    return flat_list

# Example usage
with open('fpb.json', 'r') as fin:
    json_data = json.load(fin)
    flat_list = flatten_books(json_data)
    with open('flat_fpb.json', 'w') as fout:
        json.dump(flat_list, fout, indent=2, ensure_ascii=False)
