#date: 2023-05-18T16:50:17Z
#url: https://api.github.com/gists/bb8468bc71cf0a2ab71af7b502737358
#owner: https://api.github.com/users/aspose-com-kb

import aspose.slides as slides

# Load the license
lic = slides.License()
lic.set_license("Aspose.Total.lic")

# Load the destination presentation
MainPres = slides.Presentation("Main.pptx")

# Load the presentations whose slides are to be cloned
SubPres1 = slides.Presentation("SubPres1.pptx")
SubPres2 = slides.Presentation("SubPres2.pptx")

# Iterate through all slides
for slide in SubPres1.slides:
    # Clone each slide
    MainPres.slides.add_clone(slide)

# Iterate through all slides
for slide in SubPres1.slides:
    # Clone each slide
    MainPres.slides.add_clone(slide)

MainPres.save("result.pptx", slides.export.SaveFormat.PPTX)

print("Done")
