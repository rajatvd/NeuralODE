"""Ingredient for making a ConvNet model for MNIST"""

import torch

from sacred import Ingredient
from modules import ConvNet

model_ingredient = Ingredient('model')

@model_ingredient.config
def model_config():
    """Config for model"""
    input_size = 28
    channels = [1, 32, 16]
    denses = [10]
    activation = 'relu'
    device = 'cpu'

@model_ingredient.capture
def make_model(input_size,
               channels,
               denses,
               activation,
               device,
               _log):
    """Create ConvNet model from config"""
    model = ConvNet(input_size=input_size,
                    channels=channels,
                    denses=denses,
                    activation=activation).to(device)

    params = torch.nn.utils.parameters_to_vector(model.parameters())
    num_params = len(params)
    _log.info(f"Created model with {num_params} parameters \
    on {device}")
    return model
