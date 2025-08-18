#date: 2025-08-18T17:07:45Z
#url: https://api.github.com/gists/c38c92deffe24134b5021ff6c0c99389
#owner: https://api.github.com/users/aspose-com-kb

import aspose.imaging as imaging
import aspose.pycore as apc

from aspose.imaging import Image, RasterImage, Graphics, Color, SmoothingMode, Font, FontStyle, License
from aspose.imaging.imageoptions import PngOptions
from aspose.imaging.fileformats.png import PngColorType
from aspose.imaging.sources import StreamSource
from aspose.imaging.brushes import SolidBrush

# Apply license
lic = License()
lic.set_license("license.lic")

width, height = 300, 300

# PNG options with alpha channel
options = PngOptions()
options.color_type = PngColorType.TRUECOLOR_WITH_ALPHA
options.source = StreamSource()  # in-memory image

with Image.create(options, width, height) as image:
   with apc.as_of(image, RasterImage) as raster:
       
        # Ensure internal buffers are ready before pixel writes
        if isinstance(raster, RasterImage):
            raster.cache_data()
        
        # Fill background with fully transparent pixels
        transparent = Color.from_argb(0, 255, 255, 255).to_argb()
        pixels = [transparent] * (width * height)
        raster.save_argb_32_pixels(raster.bounds, pixels)
        
        # Draw shapes/text
        g = Graphics(image)
        g.smoothing_mode = SmoothingMode.ANTI_ALIAS
        
        # Red circle in center
        g.fill_ellipse(SolidBrush(Color.red), 50, 50, 200, 200)

        # Small green circle at the corner
        g.fill_ellipse(SolidBrush(Color.green), 25, 25, 25, 25)

        # “My Logo” text in gold
        font = Font("Arial", 12, FontStyle.REGULAR)
        g.draw_string("My Logo", font, SolidBrush(Color.gold), 50, 25)

        # (Optional) dispose graphics explicitly if your version supports it
        try:
            g.dispose()
        except Exception:
            pass

        # Save PNG with transparency
        image.save("circle_transparent.png", options)
    
print("Transparent image created successfully")
