#date: 2022-08-25T17:05:55Z
#url: https://api.github.com/gists/45a9fbe9f6c247ef044be2f8dfc7430e
#owner: https://api.github.com/users/stenson

from coldtype import *

# Generic Style-to-blackrenderer.render.renderText mapping function

import tempfile
from coldtype.img.skiaimage import SkiaImage
from blackrenderer.render import renderText

def blackrender(text, style:Style):
    with tempfile.NamedTemporaryFile("wb", suffix=".png", delete=False) as tf:
        renderText(style.font.path, text, tf.name,
            backendName="skia",
            fontSize=style.fontSize,
            features=style.features,
            variations=style.variations)
    
    skimg = SkiaImage(tf.name)
    Path(tf.name).unlink()
    return skimg

# /end generic function mapping

colrv1 = "~/Type/Typeworld/color-fonts/fonts/twemoji-cff_colr_1.otf"

@renderable((1080, 540), bg=hsl(0.9, 0.3))
def test_black(r):
    return (blackrender("☕️🍹", Style(colrv1, 200
        , wdth=1
        , wght=1
        , ss01=1))
        .align(r))