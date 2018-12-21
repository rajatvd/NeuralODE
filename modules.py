"""Modules for building the CNN for MNIST"""

import torch
from torch import nn

# %%
ACTS = {
    'relu':nn.ReLU,
    'sigmoid':nn.Sigmoid,
    'tanh':nn.Tanh,}

# %%

class ConvNet(nn.Module):
    """
    CNN for MNIST using 3x3 convs and one final FC layer.
    """
    def __init__(self,
                 input_size=28,
                 channels=None,
                 denses=None,
                 activation='relu'):
        """CNN for MNIST using 3x3 convs followed by fully connected layers.
        Performs one 2x2 max pool after the first conv.

        Parameters
        ----------
        input_size : int
            Dimension of input square image (the default is 28 for MNIST).
        channels : list of ints
            List of channels of conv layers including input channels
            (the default is [1,32,32,16,8]).
        denses : list of ints
            Sequence of linear layer outputs after the conv layers
            (the default is [10]).
        activation : str
            One of 'relu', 'sigmoid' or 'tanh' (the default is 'relu').

        """
        super().__init__()
        channels = channels or [1, 32, 32, 16, 8]
        denses = denses or [10]

        act = ACTS[activation]

        convs = [nn.Conv2d(kernel_size=3, in_channels=in_ch, out_channels=out_ch)
                 for in_ch, out_ch in zip(channels[:-1], channels[1:])]

        if len(channels) <= 1:
            self.conv_net = None
            feature_count = input_size*input_size
        else:
            self.conv_net = nn.Sequential(
                convs[0],
                nn.MaxPool2d(kernel_size=2),
                act(),
                *[layer for tup in zip(convs[1:], [act() for _ in convs[1:]]) for layer in tup]
            )

            with torch.no_grad():
                test_inp = torch.randn(1, 1, input_size, input_size)
                features = self.conv_net(test_inp)
                feature_count = features.view(-1).shape[0]

        linears = [nn.Linear(in_f, out_f) for in_f, out_f in
                   zip([feature_count]+denses[:-1], denses)]

        self.dense = nn.Sequential(
            *[layer for tup in zip(linears, [act() for _ in linears]) for layer in tup][:-1]
        )


    def forward(self, input):
        if self.conv_net:
            input = self.conv_net(input)
        out = self.dense(input.view(input.shape[0], -1))
        return out

# # %%
#
# net = ConvNet()
# test_inp = torch.randn(32,1,28,28)
# out = net(test_inp)
# out.shape
# net
