#date: 2025-05-07T17:12:06Z
#url: https://api.github.com/gists/aadb96ce8b27e689bcbf6fc6b6176efa
#owner: https://api.github.com/users/aspose-com-kb

import os
import aspose.imaging as ai
import aspose.pycore as pycore
from aspose.imaging import Image, Color, TextRenderingHint, SmoothingMode
from aspose.imaging.imageoptions import PsdOptions, VectorRasterizationOptions, MultiPageOptions
from aspose.imaging.fileformats.cdr import CdrImage

# Set up Aspose license from file
lic = ai.License()
lic.set_license("license.lic")

# Define input and output file paths
input_file = "Multipage.cdr"
output_file = "result.psd"

# Open the CDR image file for processing
with Image.load(input_file) as loaded_image:
    # Convert generic Image to a more specific CdrImage instance
    corel_draw_image = pycore.as_of(loaded_image, CdrImage)

    # Prepare PSD export settings
    psd_save_options = PsdOptions()
    
    # Enable export of all pages as one layer set
    multi_page = MultiPageOptions()
    multi_page.merge_layers = True
    psd_save_options.multi_page_options = multi_page

    # Obtain default rasterization settings and cast them appropriately
    raster_opts = loaded_image.get_default_options([Color.white, loaded_image.width, loaded_image.height])
    vector_opts = pycore.as_of(raster_opts, VectorRasterizationOptions)

    # Fine-tune rendering quality
    vector_opts.text_rendering_hint = TextRenderingHint.SINGLE_BIT_PER_PIXEL
    vector_opts.smoothing_mode = SmoothingMode.NONE
    psd_save_options.vector_rasterization_options = vector_opts

    # Export the modified image to PSD format
    loaded_image.save(output_file, psd_save_options)
