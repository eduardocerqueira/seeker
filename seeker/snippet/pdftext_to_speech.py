#date: 2022-07-15T17:05:40Z
#url: https://api.github.com/gists/9baec25baf1a98867355fe750988a95f
#owner: https://api.github.com/users/Blankscreen-exe

import pyttsx3  #text to speech conversion library
import PyPDF2   #PDF manipulation library

page_to_read = 6  #page num to read
book_name = "myBook.pdf" 

#opening a pdf book
book = open(book_name , 'rb')
#reading the file
pdfreader = PyPDF2.pdffilereader(book)
#getting the number of pages
pages = pdfreader.numpages

#printing pages to stdoutput
print(pages)

#initializing speaker
speaker = pyttsx3.init()
#getting a single relevant page to read
page = pdfreader.getpage(page_to_read)
#extracting text from the page
text = page.extracttext()

#reading the extracted text
speaker.say(text)
speaker.runandwait()