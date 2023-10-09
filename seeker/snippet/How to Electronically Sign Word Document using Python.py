#date: 2023-10-09T16:56:14Z
#url: https://api.github.com/gists/b9791a72eb061d9f611f3cfd15078323
#owner: https://api.github.com/users/aspose-com-kb

import aspose.words as aw
import aspose.pydrawing as drawing
import datetime

# Load the license
wordLic = aw.License()
wordLic.set_license("License.lic")

# Create a certificate holder
certificate = aw.digitalsignatures.CertificateHolder.create("certificate.pfx", "mypass", None)

# Create digital signature options
options = aw.digitalsignatures.SignOptions()

# Set comments
options.comments = "Signing Authority Comments"

# Set signature time
options.sign_time = datetime.datetime(2023,10,9,20,0,0)

# Sign the document
aw.digitalsignatures.DigitalSignatureUtil.sign("Document.docx","SignedDocument.docx",certificate,options)

print ("Signature added to Word file successfully")
