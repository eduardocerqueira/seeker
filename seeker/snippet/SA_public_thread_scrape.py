#date: 2023-12-06T16:42:22Z
#url: https://api.github.com/gists/986b2a8c86e50379c854d759e34dbb30
#owner: https://api.github.com/users/RooneyMcNibNug

### Python script using bs4 to scrape a Something Awful forums thread. ###
#
# ! THIS WILL ONLY WORK ON THREADS THAT ARE PUBLICLY ACCESSIBLE, IE ONE'S YOU DON'T NEED TO LOG IN TO SEE !
# (I didn't want to deal with auth stuff..)
#
# Make sure you replace the thread_url variable with the link you want to scrape, same as the example in the code here.
#
# This will dump to an HTML file with decent formatting (a bit too wall-of-text atm, but I have too many other things to tend to in life).
#
# If you want to have better file naming including something like the ThreadID, do this:
# $ SA_public_thread_scrape.py && thread_id=$(sed -n 's/.*threadid=\([0-9]*\).*/\1/p' SA_public_thread_scrape.py) && mv scraped_data.html scraped_data_${thread_id}.html


import requests
from bs4 import BeautifulSoup

def scrape_page(page_url):
    response = requests.get(page_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    posts = soup.find_all('table', class_='post')

    for post in posts:
        # Extract user information
        userinfo = post.find('dl', class_='userinfo')
        try:
            # Extract 'author' (user)
            author_text = userinfo.find('dt', class_='author').get_text()
        except AttributeError:
            author_text = "Unknown"

        # Extract 'registered' date
        try:
            registered_text = userinfo.find('dd', class_='registered').get_text()
        except AttributeError:
            registered_text = "Unknown"
        
        # Format the user data with a "-" between
        user_data = f"<strong>User:</strong> {author_text} - Registered: {registered_text}<br>"

        # Extract the timestamp of the post
        timestamp = post.find('td', class_='postdate').get_text()

        # Extract the body of the post
        content = post.find('td', class_='postbody').get_text(strip=True)

        # Extract and format any images in the post, minus avatars and tagsigns and such
        images = post.find_all('img', class_=lambda x: x in ['img', 'timg', 'complete'])
        image_data = ""
        for img in images:
            image_src = img.get('src')
            if ("safs/titles" in image_src or "images/gangtags" in image_src or "images/svgs" in image_src or 
                "customtitles" in image_src or "images/avatars" in image_src or "images/newbie.gif" in image_src
                or "images/title-banned.gif" in image_src):
                continue
            image_data += f'<img src="{image_src}" alt="Image"><br>'

        # Combine the user data, post data, and images
        post_data = user_data +\
                    f"<strong>Timestamp:</strong> {timestamp}<br>"\
                    f"<strong>Content:</strong> {content}<br>" +\
                    image_data +\
                    "<hr><br>"

        scraped_data.append(post_data)

def get_next_page(current_page):
    response = requests.get(current_page)
    soup = BeautifulSoup(response.content, 'html.parser')
    next_page_link = soup.find('a', string='›')  # find an anchor element whose string content is '›'
    if next_page_link is not None:
        next_page_url = 'https://forums.somethingawful.com/' + next_page_link['href']
        return next_page_url
    else:
        return None

thread_url = 'https://forums.somethingawful.com/showthread.php?threadid=4048837'  # THE SOMETHINGAWFUL FORUM URL YOU WANT TO SCRAPE GOES HERE!
scraped_data = []

current_page = thread_url
while current_page is not None:
    scrape_page(current_page)
    current_page = get_next_page(current_page)

# Save the scraped data to an HTML file
with open('scraped_data.html', 'w', encoding='utf-8') as f:
    f.write("<html>")
    f.write("<head>")
    f.write("<title>Scraped Data</title>")
    f.write("</head>")
    f.write("<body>")
    for data in scraped_data:
        f.write(data)
    f.write("</body>")
    f.write("</html>")

print("Scraped data has been saved to scraped_data.html")