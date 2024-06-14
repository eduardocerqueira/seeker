#date: 2024-06-14T17:01:52Z
#url: https://api.github.com/gists/236a68a026682e604b0904112a300e1d
#owner: https://api.github.com/users/aspose-com-kb

import aspose.page
from aspose.page import *
from aspose.page.eps.device import *
from aspose.page.eps import *
import io


#Create output stream for PostScript document
with open("CreateEPS.ps", "wb") as out_ps_stream:
    #Create save options
    options = PsSaveOptions()
    #If you want to aassign page size other than A4, set page size in options
    options.page_size = PageConstants.get_size(PageConstants.SIZE_A4, PageConstants.ORIENTATION_PORTRAIT)
    #If you want to aassign page margins other empty, set page margins in options
    options.margins = PageConstants.get_margins(PageConstants.MARGINS_ZERO)
   
    #Set variable that indicates if resulting PostScript document will be multipaged
    multi_paged = False
   
    # Create new PS Document with one page opened
    document = PsDocument(out_ps_stream, options, multi_paged)
   
    #Close current page
    document.close_page()
    #Save the document
    document.save()