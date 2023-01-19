#date: 2023-01-19T16:52:08Z
#url: https://api.github.com/gists/7f542cdf5e051db758e462a2a8cc50fd
#owner: https://api.github.com/users/YosuaMichael

import torch
import torchvision
import random

from PIL import Image

# Image from: https://github.com/pytorch/vision/blob/main/test/assets/encode_jpeg/grace_hopper_517x606.jpg
img_path = "grace_hopper_517x606.jpg"
img_pil = Image.open(img_path)

def get_cpu_gpu_model_output_maxdiff(model_fn, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    # Use real weight, we use the DEFAULT weight
    weight_enum = torchvision.models.get_model_weights(model_fn)
    weight = weight_enum.DEFAULT

    preprocess = weight.transforms()
    x_cpu = preprocess(img_pil).unsqueeze(0).to("cpu")
    x_gpu = preprocess(img_pil).unsqueeze(0).to("cuda")

    m_cpu = model_fn(weights=weight).eval()
    m_gpu = model_fn(weights=weight).cuda().eval()

    y_cpu = m_cpu(x_cpu).squeeze(0)
    y_gpu = m_gpu(x_gpu).to("cpu").squeeze(0)

    abs_diff = torch.abs(y_gpu - y_cpu)
    max_abs_diff = torch.max(abs_diff)
    max_abs_idx = torch.argmax(abs_diff)
    max_rel_diff = torch.abs(max_abs_diff / y_cpu[max_abs_idx])
    max_val_gpu = torch.max(torch.abs(y_gpu))
    mean_val_gpu = torch.mean(torch.abs(y_gpu))
    prec = 1e-3
    pass_test = torch.allclose(y_gpu, y_cpu, atol=prec, rtol=prec)
    print(f"  [{seed}]max_abs_diff: {max_abs_diff},\tmax_rel_diff: {max_rel_diff},\tmax_val_gpu: {max_val_gpu},\tmean_val_gpu: {mean_val_gpu},\tpass_test: {pass_test}")


all_model_fns = [torchvision.models.get_model_builder(model_name) for model_name in torchvision.models.list_models(torchvision.models)]
for model_fn in all_model_fns:
    print(f"model_fn: {model_fn.__name__}")
    for seed in range(1):
        get_cpu_gpu_model_output_maxdiff(model_fn, seed)