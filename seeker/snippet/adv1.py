#date: 2023-03-22T16:54:58Z
#url: https://api.github.com/gists/36817dba4838d9ab1d300fa2491e68a9
#owner: https://api.github.com/users/corrosivelogic

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


pig_img = Image.open("pig.jpg")
preprocess = transforms.Compose([
   transforms.Resize(224),
   transforms.ToTensor(),
])
pig_tensor = preprocess(pig_img)[None,:,:,:]

plt.imshow(pig_tensor[0].numpy().transpose(1,2,0))