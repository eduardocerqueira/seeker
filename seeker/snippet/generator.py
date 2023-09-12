#date: 2023-09-12T16:57:10Z
#url: https://api.github.com/gists/b0f25abc14a268c9c49c1a4879dfabf9
#owner: https://api.github.com/users/Shaam-K

# Importing the PIL library
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import csv
 
csv_file_path = 'names.csv'

with open(csv_file_path, 'r') as file:
    csv_reader = csv.reader(file)
    data_list = []
    for row in csv_reader:
        data_list += row



print(len(data_list)) # check this number with no of names present is same as name list


for name in data_list:
    # Open an Ima
    img = Image.open('certificate.jpg')

 
    # Call draw Method to add 2D graphics in an image
    I1 = ImageDraw.Draw(img)
    
    # Custom font style and font size
    myFont = ImageFont.truetype('font.ttf', 65) # specify font



    # Add Text to an image
    I1.text((800, 675), name.title(), font=myFont, fill =(123, 76, 4),anchor='ms') 
    
    # 800, 675 are x and y coordinates (i think), fill takes rgb value
    
    img.save(f'certificates_cyber_event/{name.upper()}.jpg', 'JPEG')