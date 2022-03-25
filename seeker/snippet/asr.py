#date: 2022-03-25T17:04:57Z
#url: https://api.github.com/gists/e70a19504338177d7742dc4f3a1b9894
#owner: https://api.github.com/users/philschmid

from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serializers import DataSerializer
import sagemaker

role = sagemaker.get_execution_role()
# Hub Model configuration. https://huggingface.co/models

hub = {
	'HF_MODEL_ID':'facebook/wav2vec2-base-960h',
	'HF_TASK':'automatic-speech-recognition'
}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
	transformers_version='4.17',
	pytorch_version='1.10',
	py_version='py38',
	env=hub,
	role=role, 
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
	initial_instance_count=1, # number of instances
	instance_type='ml.m5.xlarge' # ec2 instance type
	serializer=DataSerializer(content_type="audio/wave") # serializer for mime-type
)

# send request with file_path
transcription = predictor.predict("path/to/interview.wav")
