#date: 2023-11-17T16:40:23Z
#url: https://api.github.com/gists/5ad525af3a462935a99cff1944587b21
#owner: https://api.github.com/users/aspose-com-kb

import aspose.words
from aspose.words import * 
from datetime import datetime

# Create a Document class object 
docA = Document("Compare1.pdf") 
docB = Document("Compare2.pdf")

# Create CompareOptions class object
options = comparing.CompareOptions()
options.target = comparing.ComparisonTargetType.NEW

# Compare Word documents
docA.compare(docB, "Author", datetime.now(), options)

# Save the document
docA.save("Comparison_Output.pdf")

print ("PDF files compared successfully")