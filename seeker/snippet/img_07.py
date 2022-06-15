#date: 2022-06-15T16:50:02Z
#url: https://api.github.com/gists/9bc7fb6c6ee9cf99fe61e69d9a2453c1
#owner: https://api.github.com/users/SoftSAR

import SimpleITK as sitk
img = sitk.ReadeImage('my_input.png')
sitk.WriteImage(img, 'my_output.jpg')