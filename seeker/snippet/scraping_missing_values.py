#date: 2022-03-10T17:06:10Z
#url: https://api.github.com/gists/741f8f7997ca15fe7e39f9d3c9dcb7bb
#owner: https://api.github.com/users/JLFDataScience

data = {
    'name': [],
    'urls': [],
    'position': [],
    'area': [],
    'year': [],
    'city': []
    }

for tag in soup.find_all('article', 'box becari buscador'):
    name = tag.find('a')
    urls = tag.find('a')
    position = tag.find('span', 'posicio')
    area = tag.find('p', 'disciplina')
    year = tag.find('p', 'formacio')
    city = tag.find('span', 'ciutat')
    data['name'].append(name.get_text().strip() if name else 'N/A')
    data['urls'].append(urls['href'].strip() if urls else 'N/A')
    data['position'].append(position.get_text().strip() if position else 'N/A')
    data['area'].append(area.get_text().strip() \
                        .replace('Academic discipline:\n\t\t\t\t\t\t\t\t', '') if area else 'N/A')
    data['year'].append(''.join(re.findall('\d+', year.get_text().strip())) if year else 'N/A')
    data['city'].append(city.get_text().strip() if city else 'N/A')