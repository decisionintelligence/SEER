import torch
import torch.nn as nn
from ..layers.Autoformer_EncDec import series_decomp


class Linear_extractor(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Linear_extractor, self).__init__()

        self.value_embedding = nn.Linear(configs.patch_len, configs.d_model, bias=False)


    def forward(self, x_enc):
        x = self.value_embedding(x_enc)
        return x

