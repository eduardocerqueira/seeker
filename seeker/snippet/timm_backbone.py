#date: 2022-07-30T19:20:56Z
#url: https://api.github.com/gists/03a2fafdeaad066b1d71295c99352a55
#owner: https://api.github.com/users/curt-tigges

import timm
import torch.nn as nn


class TimmBackbone(nn.Module):
    """Specified timm model without pooling or classification head"""

    def __init__(self, model_name):
        """Downloads and instantiates pretrained model

        Args:
            model_name (str): Name of model to instantiate.
        """
        super().__init__()

        # Creating the model in this way produces unpooled, unclassified features
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=0, global_pool=""
        )

    def forward(self, x):
        """Passes batch through backbone

        Args:
            x (Tensor): Batch tensor consisting of batch of processed images.

        Returns:
            Tensor: Unpooled, unclassified features from image model.
        """

        out = self.model(x)

        return out