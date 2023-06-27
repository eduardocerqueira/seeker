#date: 2023-06-27T16:56:32Z
#url: https://api.github.com/gists/3c3c7fcbdf598456ce2d8528a5527d2d
#owner: https://api.github.com/users/jrkerns

import pylinac


class CatphanFOV(pylinac.CatPhan504):

    @property
    def mm_per_pixel(self) -> float:
        return super().mm_per_pixel * 0.81


cbct = CatphanFOV(r"C:\Users\jkern\Downloads\pylinaczip - Matthew Georgesen")
cbct.analyze()
cbct.plot_analyzed_image()