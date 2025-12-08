#date: 2025-12-08T17:05:07Z
#url: https://api.github.com/gists/1a1ee3ddbce2b1362b673b2fa7d1feb0
#owner: https://api.github.com/users/jffng

import re
import csv
from datetime import datetime
from collections import defaultdict

# Read the file
with open('/mnt/user-data/uploads/My_Clippings___Dec_8_2025.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# Split by the separator
entries = content.split('==========')

# Dictionary to store book data
books = defaultdict(lambda: {
    'author': '',
    'first_date': None,
    'highlights': []
})

# Process each entry
for entry in entries:
    entry = entry.strip()
    if not entry:
        continue
    
    lines = entry.split('\n')
    if len(lines) < 2:
        continue
    
    # First line contains book title and author
    title_line = lines[0].strip()
    
    # Extract title and author
    if '(' in title_line and ')' in title_line:
        last_paren = title_line.rfind('(')
        title = title_line[:last_paren].strip()
        author = title_line[last_paren+1:title_line.rfind(')')].strip()
    else:
        title = title_line
        author = ''
    
    # Clean up title
    if ' -- ' in title:
        title = title.split(' -- ')[0].strip()
    if title.startswith('﻿'):
        title = title[1:]
    
    # Second line contains metadata
    metadata_line = lines[1] if len(lines) > 1 else ''
    
    # Extract date
    date_match = re.search(r'Added on (.+?)$', metadata_line)
    if date_match:
        date_str = date_match.group(1).strip()
        try:
            date_obj = datetime.strptime(date_str, '%A, %B %d, %Y %I:%M:%S %p')
        except:
            try:
                date_obj = datetime.strptime(date_str, '%A, %B %d, %Y')
            except:
                date_obj = None
    else:
        date_obj = None
    
    # Extract highlight text
    highlight_text = ''
    for i in range(2, len(lines)):
        if lines[i].strip() and not lines[i].startswith('-'):
            highlight_text += lines[i].strip() + ' '
    
    highlight_text = highlight_text.strip()
    
    # Filter out very short highlights
    if highlight_text and len(highlight_text.split()) >= 10:
        if not books[title]['author']:
            books[title]['author'] = author
        
        if date_obj:
            if not books[title]['first_date'] or date_obj < books[title]['first_date']:
                books[title]['first_date'] = date_obj
        
        books[title]['highlights'].append(highlight_text)

# Theme inference
def infer_themes(title, author, highlights):
    title_lower = title.lower()
    author_lower = author.lower()
    all_text = (title + ' ' + author + ' ' + ' '.join(highlights[:5])).lower()
    
    # Specific book matching
    if 'stay true' in title_lower:
        return 'Identity, Friendship, Loss'
    elif 'brothers karamazov' in title_lower or 'dostoevsky' in author_lower:
        return 'Philosophy, Religion, Literature'
    elif 'no longer human' in title_lower or 'dazai' in author_lower:
        return 'Depression, Identity, Alienation'
    elif 'breasts and eggs' in title_lower or 'kawakami' in author_lower:
        return 'Identity, Gender, Body, Writing'
    elif 'capital' in title_lower and 'twenty' in title_lower:
        return 'Economics, Capitalism, Inequality'
    elif 'energy flash' in title_lower or 'rave' in title_lower:
        return 'Rave, Music, Sound, Culture'
    elif 'racial melancholia' in title_lower or ('eng' in author_lower and 'han' in author_lower):
        return 'Race, Psychoanalysis, Identity, Theory'
    elif 'haraway' in author_lower or 'staying with the trouble' in title_lower:
        return 'Theory, Ecology, Posthumanism, Feminism'
    elif 'health and safety' in title_lower:
        return 'Work, Capitalism, Mental Health'
    elif 'hold everything dear' in title_lower or 'berger' in author_lower:
        return 'Politics, Art, Resistance'
    elif 'parable of the sower' in title_lower or 'butler' in author_lower:
        return 'Afrofuturism, Climate, Dystopia, Race'
    elif 'rejection' in title_lower:
        return 'Writing, Practice, Publishing'
    elif 'silence' in title_lower and 'endo' in author_lower:
        return 'Religion, Philosophy, Colonialism'
    elif 'emperor of gladness' in title_lower:
        return 'Poetry, Loss, Grief, Literature'
    elif 'derivative' in title_lower:
        return 'Media, Theory, Technology'
    elif 'free us' in title_lower or 'abolition' in all_text:
        return 'Abolition, Politics, Justice, Praxis'
    elif 'room of one' in title_lower or 'woolf' in author_lower:
        return 'Feminism, Writing, Gender, Literature'
    else:
        return 'Literature, Life'

# Create CSV data
csv_data = []
for title in sorted(books.keys()):
    book = books[title]
    
    # Skip if no highlights
    if not book['highlights']:
        continue
    
    # Format date
    date_str = book['first_date'].strftime('%Y-%m-%d') if book['first_date'] else ''
    
    # Infer themes
    themes = infer_themes(title, book['author'], book['highlights'])
    
    # Combine all citations with double line breaks
    all_citations = '\n\n'.join(book['highlights'])
    
    csv_data.append({
        'Name': title,
        'Author': book['author'],
        'Date (Initial)': date_str,
        'Link': '',
        'Type': 'book',
        'Referrer': 'Self',
        'Themes': themes,
        'Citation': all_citations
    })

# Write to CSV
output_file = '/home/claude/reading_highlights.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['Name', 'Author', 'Date (Initial)', 'Link', 'Type', 'Referrer', 'Themes', 'Citation']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_data)

print(f"CSV created with {len(csv_data)} books")
print(f"Output file: {output_file}")

EOF
