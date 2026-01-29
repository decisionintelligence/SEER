import torch
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self, config):
        super(encoder, self).__init__()
        input_size = config.patch_len
        num_experts = config.num_experts

        self.distribution_fit = nn.Linear(input_size, num_experts, bias=False)

    def forward(self, x):
        out = self.distribution_fit(x)
        return out
