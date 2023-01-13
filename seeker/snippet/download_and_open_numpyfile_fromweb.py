#date: 2023-01-13T16:58:55Z
#url: https://api.github.com/gists/3c5e2d8b27a8e9a2a0fbc6611a4376d9
#owner: https://api.github.com/users/kbn-gh

import requests
import io
import numpy as np

# Wrong link please change WEB_LINK
WEB_LINK = 'https://biolinuxtest.wustl.edu/test_neuron.npy'
response = requests.get(WEB_LINK)
neurons = np.load(io.BytesIO(response.content), allow_pickle=True)