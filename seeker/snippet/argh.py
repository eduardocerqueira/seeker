#date: 2023-07-24T17:05:14Z
#url: https://api.github.com/gists/0b25d37a9207c588c2680c9e548b6ba7
#owner: https://api.github.com/users/seanpaulbradley

import os
import torch
import torch.nn as nn
import numpy as np
from thingsvision import get_extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader
from thingsvision.core.extraction import center_features
import tensorflow as tf
import keras
import torchvision #
from typing import Any, Dict, List, Optional, Union
import harmonization
from harmonization.models import (load_EfficientNetB0, load_LeViT_small,
                                  load_ResNet50, load_tiny_ConvNeXT,
                                  load_tiny_MaxViT, load_VGG16, load_ViT_B16)
from thingsvision.custom_models.custom import Custom

# Model configs
configs = {
    'vgg16': {'model_name': 'vgg16', 'module_name': 'classifier.4', 'source':'torchvision', 'variant': None},
    'resnet50': {'model_name': 'resnet50', 'module_name': 'avgpool', 'source':'torchvision', 'variant': None},
    'resnet101': {'model_name': 'resnet101', 'module_name': 'avgpool', 'source':'torchvision', 'variant': None},
    'cornet': {'model_name': 'cornet_s', 'module_name': 'decoder.flatten', 'source':'custom', 'variant': None},
    'densenet': {'model_name': 'densenet201', 'module_name': 'features.norm5', 'source':'torchvision', 'variant': None},
    'mobilenet': {'model_name': 'mobilenet_v3_large', 'module_name': 'classifier.2', 'source':'torchvision', 'variant': None},
    'mnasnet': {'model_name': 'mnasnet_1_3', 'module_name': 'classifier.0', 'source':'torchvision', 'variant': None},
    'inception': {'model_name': 'inception_v3', 'module_name': 'avgpool', 'source':'torchvision', 'variant': None},
    'efficientnet': {'model_name': 'efficientnet_b0', 'module_name': 'classifier.0', 'source':'torchvision', 'variant': None},
    'resnext': {'model_name': 'resnext50_32x4d', 'module_name': 'avgpool', 'source':'torchvision', 'variant': None},
    'alexnet': {'model_name': 'alexnet', 'module_name': 'features.classifier.5', 'source':'torchvision', 'variant': None},
    'vit-l-32': {'model_name': 'vit-l-32', 'module_name': 'ln', 'source':'torchvision', 'variant': None},
    'clip': {'model_name': 'clip', 'module_name': 'visual','source':'custom', 'variant': 'ViT-B/32'},
    'harm-RN50': {'model_name': 'Harmonization', 'module_name': 'avgpool','source':'custom', 'variant': 'RN50'},
    'harm-vgg16': {'model_name': 'Harmonization', 'module_name': 'classifier.4','source':'custom', 'variant': 'vgg16'},
    'harm-effnetb0': {'model_name': 'Harmonization', 'module_name': 'classifier.0','source':'custom', 'variant': 'EfficientNetB0'},
    'harm-t_covnext': {'model_name': 'Harmonization', 'module_name': 'head_ln','source':'custom', 'variant': 'tiny_convNeXT'},
    'harm-t_maxvit': {'model_name': 'Harmonization', 'module_name': 'features[0][0]','source':'custom', 'variant': 'tiny_MaxViT'},
    'harm-levit': {'model_name': 'Harmonization', 'module_name': 'flatten_1[0][0]','source':'custom', 'variant': 'LeViT_small'},  
}

template = """
# Imports
import os
import torch
import torch.nn as nn
import numpy as np
from thingsvision import get_extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader
from thingsvision.core.extraction import center_features
import torchvision #
from typing import Any, Dict, List, Optional, Union
import tensorflow as tf
import keras

image_dir = 'example_images'
output_dir = 'torchvision_output/{model_name}_{module_name}'
mounted_dir = '/data/bradleysp/thingsvision/'
full_image_path = os.path.join(mounted_dir, image_dir)
full_output_path = os.path.join(mounted_dir, output_dir)

def extract_features(
    extractor: Any,
    module_name: str,
    image_path: str,
    out_path: str,
    batch_size: int,
    flatten_activations: bool,
    apply_center_crop: bool,
    class_names: Optional[List[str]]=None,
    file_names: Optional[List[str]]=None,
) -> Union[np.ndarray, torch.Tensor]:
    dataset = ImageDataset(
        root=image_path,
        out_path=out_path,
        backend=extractor.get_backend(),
        transforms=extractor.get_transformations(apply_center_crop=apply_center_crop, resize_dim=256, crop_dim=224),
        class_names=class_names,
        file_names=file_names,
    )
    batches = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        backend=extractor.get_backend(),
        )
    features = extractor.extract_features(
        batches=batches,
        module_name=module_name,
        flatten_acts=flatten_activations,
        output_type="ndarray", # or "tensor" (only applicable to PyTorch models)
    )
    return features

model_name = '{model_name}'
print(model_name)
module_name = '{module_name}'
print(module_name)
source = '{source}'
print(source)
variant = {variant}
print(variant)

pretrained = True
batch_size = 64
apply_center_crop = False
flatten_activations = True
class_names = None  
file_names = None
file_format = "npy"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if variant:
    extractor = get_extractor(model_name=model_name, 
                            source=source, 
                            pretrained=pretrained, 
                            device=device
                            model_parameters={'variant': variant}
                            )
else:
    extractor = get_extractor(model_name=model_name, 
                        source=source, 
                        pretrained=pretrained, 
                        device=device
                        )
                        
features = extract_features(
extractor=extractor,
module_name=module_name,
image_path=full_image_path,
out_path=full_output_path,
batch_size=batch_size,
flatten_activations=flatten_activations,
apply_center_crop=apply_center_crop,
class_names=class_names,
file_names=file_names,
)

save_features(
features,
out_path=output_dir,
file_format='npy'
)
"""

for model in configs:

  model_name = configs[model]['model_name']
  module_name = configs[model]['module_name']
  source = configs[model]['source']
  variant = configs[model]['variant']
  print(configs)
  print('###### MODEL NAME', model_name, '#######')
  print(variant)
  

  print("Keys:", model_name, module_name, source, variant)

  script = template.format(
    model_name=model_name,
    module_name=module_name,
    source=source,
    variant=variant
  )

  with open(f'{model_name}_script.py', 'w') as f:
    f.write(script)

#Yields error:

# ###### MODEL NAME vgg16 #######
#None
#Keys: vgg16 classifier.4 torchvision None
#Traceback (most recent call last):
#  File "/gpfs/gsfs12/users/bradleysp/thingsvision/tv_batches3.py", line 161, in <module>
#    script = template.format(
#KeyError: "'variant'"
