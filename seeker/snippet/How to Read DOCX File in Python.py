#date: 2022-04-11T17:16:05Z
#url: https://api.github.com/gists/7fbb23599da05ed9b7b3656b330391e0
#owner: https://api.github.com/users/aspose-com-kb

import aspose.words as aw

# Initialize the license to avoid trial version limitations 
# while reading the word file in python
editWordLicense = aw.License()
editWordLicense.set_license("Aspose.Word.lic")

# Load the source document that needs to be read
docToRead = aw.Document("input.docx")

# Read all the contents from the node types paragraph
for paragraph in docToRead.get_child_nodes(aw.NodeType.PARAGRAPH, True) :    
    paragraph = paragraph.as_paragraph()
    print(paragraph.to_string(aw.SaveFormat.TEXT))