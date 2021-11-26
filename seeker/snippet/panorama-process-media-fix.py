#date: 2021-11-26T17:12:44Z
#url: https://api.github.com/gists/fc80b6a2993de798492296e593d9f717
#owner: https://api.github.com/users/mrtj

def process_media(self, stream):
    image_data, ratio = self.preprocess(stream.image, self.MODEL_INPUT_SIZE)
    inference_results = self.call(
        {self.MODEL_INPUT_NAME: image_data}, self.MODEL_NODE
    )
    self.process_results(inference_results, stream, ratio)