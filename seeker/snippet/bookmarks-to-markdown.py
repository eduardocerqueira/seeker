#date: 2024-03-26T17:09:36Z
#url: https://api.github.com/gists/948513b2d2addddfede53159114a2d09
#owner: https://api.github.com/users/EdoardoTosin

import json

def escape_markdown_chars(text):
    escaped_text = ""
    for char in text:
        if char in ["\\", "`", "*", "_", "{", "}", "[", "]", "<", ">", "(", ")", "#", "+", "-", ".", "!", "|"]:
            escaped_text += "\\" + char
        else:
            escaped_text += char
    return escaped_text

def convert_bookmarks_to_markdown(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        bookmarks_data = json.load(file)
    
    def process_bookmarks(bookmarks, level=0):
        markdown = ''
        for bookmark in bookmarks:
            if 'type' in bookmark and bookmark['type'] == 'text/x-moz-place':
                title = escape_markdown_chars(bookmark['title'])
                url = bookmark['uri']
                markdown += '    ' * level + f"- [{title}]({url})\n"
            elif 'type' in bookmark and bookmark['type'] == 'text/x-moz-place-container':
                title = escape_markdown_chars(bookmark['title'])
                markdown += '    ' * level + f"- {title}\n"
                markdown += process_bookmarks(bookmark['children'], level + 1)
        return markdown
    
    bookmarks = bookmarks_data['children'][1]['children']
    markdown_output = process_bookmarks(bookmarks)
    
    with open('bookmarks.md', 'w', encoding='utf-8') as file:
        file.write(markdown_output)

# Replace 'path/to/bookmarks.json' with the actual path to your Firefox bookmarks JSON file
convert_bookmarks_to_markdown('path/to/bookmarks.json')