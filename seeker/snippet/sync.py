#date: 2023-04-03T16:54:20Z
#url: https://api.github.com/gists/0f77c9261376109312badcde82a6c1a2
#owner: https://api.github.com/users/caseydm

import csv

from app import db

with open("publishers_image_thumbnail_urls.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        publisher_id = int(row[0].replace("https://openalex.org/P", ""))
        image_thumbnail_url = row[2]
        db.session.execute('UPDATE mid.publisher SET image_thumbnail_url = :image_thumbnail_url WHERE publisher_id = :publisher_id', {"image_thumbnail_url": image_thumbnail_url, "publisher_id": publisher_id})
        db.session.commit()
        print(f"Updated {publisher_id} with {image_thumbnail_url}")
