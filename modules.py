"""Modules for building the CNN for MNIST"""

import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint

# %%
ACTS = {
    'relu':nn.ReLU,
    'sigmoid':nn.Sigmoid,
    'tanh':nn.Tanh,}

# %%

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

# %%

class ODEfunc(nn.Module):

    def __init__(self, dim, act='relu'):
        super().__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(dim, dim)
        self.norm2 = norm(dim)
        self.conv2 = conv3x3(dim, dim)
        self.norm3 = norm(dim)
        self.conv3 = conv3x3(dim, dim)
        self.norm4 = norm(dim)
        self.conv4 = conv3x3(dim, dim)
        self.norm5 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm4(out)
        out = self.relu(out)
        out = self.conv4(out)

        out = self.norm5(out)
        return out

# %%

class ConvODEfunc(nn.Module):
    """Two convolution network for an ODE function.
    Inputs and outputs are the same size.

    Parameters
    ----------
    dim : int
        Number of channels in input (and output).
    act : string
        Activation function. One of relu, sigmoid or tanh (the default is 'relu').
    """


    def __init__(self, dim, act='relu'):
        super().__init__()
        self.act = ACTS[act]()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(dim)
        self.conv3 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm3 = nn.BatchNorm2d(dim)
        self.conv4 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm4 = nn.BatchNorm2d(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(x)
        out = self.norm2(out)
        out = self.act(out)
        out = self.conv3(x)
        out = self.norm3(out)
        out = self.act(out)
        out = self.conv4(out)
        out = self.norm4(out)
        return out

# %%

class ODEBlock(nn.Module):
    """Short summary.

    Parameters
    ----------
    odefunc : nn.Module
        An nn.Module which has an nfe attribute and has same input and output
        sizes.

    rtol: float
        Relative tolerance for ODE evaluations. Default 1e-3

    atol: float
        Absolute tolerance for ODE evaluations. Default 1e-3

    Forward takes x and t as inputs, and returns the final state
    at the given input time points t. Default t is [0, 1]
    self.outputs contains all the outputs at each time step.
    """

    def __init__(self, odefunc, rtol=1e-3, atol=1e-3):
        super().__init__()
        self.odefunc = odefunc
        self.t = torch.tensor([0, 1]).float()
        self.outputs = None
        self.rtol = rtol
        self.atol = atol

    def forward(self, x, t=None):
        if t is None:
            times = self.t
        else:
            times = t
        self.outputs = odeint(self.odefunc,
                              x,
                              times,
                              rtol=self.rtol,
                              atol=self.atol)
        return self.outputs[1]
    @property
    def nfe(self):
        """Number of function evaluations"""
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

# # %%
# import imageio
# device = 'cuda'
# # %%
# f = ConvODEfunc(3, act='relu').to(device).eval()
# nn.utils.parameters_to_vector(f.parameters()).shape
#
# with torch.no_grad():
#     inp = torch.randn(1,3,224,224).to(device)
#     f(0, inp).shape
#     t = torch.linspace(0,500,100).to(device)
#     odenet = ODEBlock(f, rtol=1e-3, atol=1e-3).to(device).eval()
#     odenet(inp, t).shape
#     print(odenet.nfe)
#     print(odenet.outputs.shape)
#
# with torch.no_grad():
#     ims = odenet.outputs
#     ims = torch.sigmoid(ims)
#     ims = ims.squeeze().cpu().detach().numpy().transpose([0,2,3,1])
#     ims.shape
#     imageio.mimwrite("test.gif", ims, duration=0.05)
#
# # %%
#
#     with torch.no_grad():
#     ims2 = []
#     inp = torch.randn(1,3,224,224)
#     for i in range(100):
#         inp += f(0,inp)
#         print(inp.std())
#         ims2.append(torch.sigmoid(inp).squeeze().detach().numpy().transpose([1,2,0]))
#
#     imageio.mimwrite("conving.gif", ims2)
#
# %%timeit
# torch.eig(torch.randn(784,784))

# %%

class ODEnet(nn.Module):
    """ODE net for classifying images.

    Performs one downsampling conv + fractional max pool. Then applies the ode
    network, followed by a final fully connected layer after flattening.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    state_channels : int
        Number of channels in the state of the ODE. Output channels of the first
        downsampling conv.
    state_size : int
        Height(=width) of the state of ODE. Output size of first downsampling.
    output_size : int
        Number of output classes (the default is 10).
    act : string
        Activation for the odefunc. (relu, sigmoid or tanh) (the default is 'relu').
    tol : float
        Relative and absolute tolerance for ODE evaluations (the default is 1e-3).
    """
    def __init__(self,
                 in_channels,
                 state_channels,
                 state_size,
                 output_size=10,
                 act='relu',
                 tol=1e-3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, state_channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(state_channels)
        self.pool = nn.FractionalMaxPool2d(2, output_size=state_size)
        self.odefunc = ODEfunc(state_channels, act=act)
        self.odeblock = ODEBlock(self.odefunc, rtol=tol, atol=tol)
        self.fc = nn.Linear(state_size*state_size*state_channels,
                            output_size)

    def forward(self, x, t=None):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.pool(out)
        out = self.odeblock(out, t)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out

# # %%
#
# odenet = ODEnet(1, 16, 7, tol=1e-6)
#
# test_in = torch.randn(32,1,28,28)
# test_out = odenet(test_in)
# nn.utils.parameters_to_vector(odenet.parameters()).shape
# t = torch.linspace(0,1,100)
# test_out = odenet(test_in, t=t)
#
# import pytorch_utils.sacred_trainer as st
# loader_test = ((torch.randn(32,1,28,28), torch.randint(0,10, (32,))) for _ in range(32))
# odenet.load_state_dict(torch.load("ODEMnistClassification\\12\\epoch001_24-12_0026_.statedict.pkl"))
# import training_functions as tf
# tf.validate(odenet.cpu(), loader_test)
# odenet.train()
