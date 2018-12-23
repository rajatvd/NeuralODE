"""Ingredient for making a ODEnet model for MNIST"""

import torch

from sacred import Ingredient
from modules import ODEnet

model_ingredient = Ingredient('model')

@model_ingredient.config
def model_config():
    """Config for model"""
    in_channels = 1
    state_channels = 16
    state_size = 7
    output_size = 10
    act = 'relu'
    tol = 1e-3
    device = 'cpu'

@model_ingredient.capture
def make_model(in_channels,
               state_channels,
               state_size,
               output_size,
               act,
               tol,
               device,
               _log):
    """Create ODEnet model from config"""
    model = ODEnet(in_channels,
                   state_channels,
                   state_size,
                   output_size=output_size,
                   act=act,
                   tol=tol).to(device)

    params = torch.nn.utils.parameters_to_vector(model.parameters())
    num_params = len(params)
    _log.info(f"Created ODEnet model with {num_params} parameters \
    on {device}")

    ode_params = torch.nn.utils.parameters_to_vector(
        model.odefunc.parameters()).shape[0]
    _log.info(f"ODE function has {ode_params} parameters")
    return model
