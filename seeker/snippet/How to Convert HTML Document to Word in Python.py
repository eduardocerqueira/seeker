#date: 2023-10-02T17:07:12Z
#url: https://api.github.com/gists/094b02cbedd0fc1a525b554414f66303
#owner: https://api.github.com/users/aspose-com-kb

import aspose.words as aw
import aspose.pydrawing as drawing

# Load the license
wordLic = aw.License()
wordLic.set_license("License.lic")

# Load the HTML
htmlDoc = aw.Document("Sample.html")

# Append some text
htmlDoc.first_section.body.first_paragraph.append_child(aw.Run(htmlDoc, "This text is added for demonstration"))

# Save the loaded HTML document as DOCX
htmlDoc.save("output.docx", aw.SaveFormat.DOCX)

print ("HTML to Word file converted successfully")