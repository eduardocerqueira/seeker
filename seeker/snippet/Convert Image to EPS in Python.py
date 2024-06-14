#date: 2024-06-14T17:03:17Z
#url: https://api.github.com/gists/a8518e1b334b07944a1297a353a1bd7a
#owner: https://api.github.com/users/aspose-com-kb

import aspose.page
from aspose.page.eps import *
from aspose.page.eps.device import *
from aspose.page.eps.xmp import *

# Create default options
options = PsSaveOptions()
       
# Save image to EPS format
PsDocument.save_image_as_eps("sample_image.jpg", "output1.eps", options)