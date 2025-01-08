#date: 2025-01-08T16:44:49Z
#url: https://api.github.com/gists/c026d653f7cd7106f0a393b783689047
#owner: https://api.github.com/users/aspose-com-kb

import aspose.ocr as api
from aspose.ocr import License

# Instantiate a license
license = License()
license.set_license("License.lic")

extractTextFromImage = api.AsposeOcr()
imageDatas = api.OcrInput(api.InputType.DIRECTORY)
imageDatas.add("/Users/myuser/Images/")
textExtractedFromImage = extractTextFromImage.recognize(imageDatas)
length = textExtractedFromImage.length
for i in range(length):
    print(textExtractedFromImage[i].recognition_text)