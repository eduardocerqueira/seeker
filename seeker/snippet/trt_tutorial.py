#date: 2023-04-26T17:08:33Z
#url: https://api.github.com/gists/993189c0614b036a19ff62f0082e045f
#owner: https://api.github.com/users/YHRen

# https://pytorch.org/TensorRT/_notebooks/Resnet50-example.html
# nvcr.io/nvidia/pytorch:21.02-py3

import torch
import torchvision

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

# resnet50_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

from torchvision.models import resnet50, ResNet50_Weights
resnet50_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
resnet50_model.eval()


from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

fig, axes = plt.subplots(nrows=2, ncols=2)

for i in range(4):
    img_path = './data/img%d.JPG'%i
    img = Image.open(img_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img)
    plt.subplot(2,2,i+1)
    plt.imshow(img)
    plt.axis('off')

# loading labels
with open("./data/imagenet_class_index.json") as json_file:
    d = json.load(json_file)

## define utility functions

import numpy as np
import time
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def rn50_preprocess():
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess

# decode the results into ([predicted class, description], probability)
def predict(img_path, model):
    img = Image.open(img_path)
    preprocess = rn50_preprocess()
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        sm_output = torch.nn.functional.softmax(output[0], dim=0)

    ind = torch.argmax(sm_output)
    return d[str(ind.item())], sm_output[ind] #([predicted class, description], probability)

def benchmark(model, input_shape=(1024, 1, 224, 224),
        dtype='fp32', nwarmup=50, nruns=10000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype=='fp16':
        input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        start_time = time.time()
        for i in range(1, nruns+1):
            features = model(input_data)
        end_time = time.time()
        torch.cuda.synchronize()
        timings = (end_time - start_time)/nruns

    print("Input shape:", input_data.size())
    print("Output features size:", features.size())
    print('Average batch time: %.2f ms'%(timings*1000))

# show predictions

for i in range(4):
    img_path = './data/img%d.JPG'%i
    img = Image.open(img_path)

    pred, prob = predict(img_path, resnet50_model)
    print('{} - Predicted: {}, Probablility: {}'.format(img_path, pred, prob))

    plt.subplot(2,2,i+1)
    plt.imshow(img);
    plt.axis('off');
    plt.title(pred[1])

# Model benchmark without Torch-TensorRT
model = resnet50_model.eval().to("cuda")
benchmark(model, input_shape=(128, 3, 224, 224), nruns=100)
# 75.75 ms

# Torchscript
trace_model_fp32 = torch.jit.trace(model,
        torch.rand(128, 3, 224, 224, dtype=torch.float32, device="cuda"))
benchmark(trace_model_fp32, input_shape=(128, 3, 224, 224), nruns=100)
# 54.86 ms

trace_model_fp16 = torch.jit.trace(model,
        torch.rand(128, 3, 224, 224, dtype=torch.half, device="cuda"))
benchmark(trace_model_fp16, input_shape=(128, 3, 224, 224), nruns=100)
# TODO:

# Model benchmark with TRT
import torch_tensorrt
# The compiled module will have precision as specified by "op_precision".
# Here, it will have FP32 precision.
trt_model_fp32 = torch_tensorrt.compile(
        model,
        inputs = [torch_tensorrt.Input((128, 3, 224, 224), dtype=torch.float32)],
        enabled_precisions = torch.float32, # Run with FP32
        workspace_size = 1 << 22
)
# Obtain the average time taken by a batch of input
benchmark(trt_model_fp32, input_shape=(128, 3, 224, 224), nruns=100)
# 36.04 ms


# *Half Precision*
# The compiled module will have precision as specified by "op_precision".
# Here, it will have FP16 precision.
trt_model_fp16 = torch_tensorrt.compile(
        model,
        inputs = [torch_tensorrt.Input((128, 3, 224, 224), dtype=torch.half)],
        enabled_precisions = {torch.half}, # Run with FP32
        workspace_size = 1 << 22
)
# Obtain the average time taken by a batch of input
benchmark(trt_model_fp16, input_shape=(128, 3, 224, 224), dtype='fp16', nruns=100)
# 12.14 ms


# *Quantized Model*

testing_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(224, 224)), # pretend to be larger images
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
)

testing_dataloader = torch.utils.data.DataLoader(
    testing_dataset, drop_last=True, batch_size=128, shuffle=False,
    num_workers=1)
calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
    testing_dataloader,
    cache_file="./calibration.cache",
    use_cache=False,
    algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
    device=torch.device("cuda:0"),
)
trt_model_int8 = torch_tensorrt.compile(
        trace_model_fp16, # use traced model
        inputs=[torch_tensorrt.Input((128, 3, 224, 224), dtype=torch.half)],
        enabled_precisions={torch.int8},
        calibrator=calibrator,
        device={"device_type": torch_tensorrt.DeviceType.GPU,
                "gpu_id": 0,
                "dla_core": 0,
                "allow_gpu_fallback": False,
                "disable_tf32": False}
)
benchmark(trt_model_int8, input_shape=(128, 3, 224, 224), dtype="fp16")
# 7.05 ms
