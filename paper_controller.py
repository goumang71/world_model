"""CarRacing action mapping from Ha and Schmidhuber's author implementation.

This changes only the 288-to-3 controller. The local VAE/MDN-RNN and their
96-pixel image preprocessing remain the existing reproduction.
"""
import torch
from torch import nn


class PaperController(nn.Module):
    def __init__(self, state_dim=288, action_dim=3):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(state_dim, action_dim))

    def forward(self, x):
        raw = self.model(x)
        action = torch.tanh(raw)
        return torch.cat((action[:, 0:1], (action[:, 1:2] + 1.0) / 2.0,
                          action[:, 2:3].clamp(0.0, 1.0)), dim=1)


def warm_start_from_local(controller, local_state):
    """Match old gas exactly; brake has no exact global map between activations."""
    controller.load_state_dict(local_state)
    with torch.no_grad():
        controller.model[0].weight[1].mul_(0.5)
        controller.model[0].bias[1].mul_(0.5)
    return controller
