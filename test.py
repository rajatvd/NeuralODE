"""Test a ODENet on the MNIST dataset."""
# %%
import torch
from torch import nn
from sacred import Experiment
from pytorch_utils.sacred_utils import read_config, get_model_path, import_source
from training_functions import validate


# %%
ex = Experiment('test_mnist')

@ex.config
def input_config():
    """Parameters for sampling using the given model"""
    run_dir = 'runs/ODEMnistClassification/8'
    epoch = 'latest'
    device = 'cpu'
    min_end_time = 10
    max_end_time = 100
    tol = 1e-3

@ex.automain
def main(run_dir,
         epoch,
         device,
         min_end_time,
         max_end_time,
         tol,
         _log):

    config = read_config(run_dir)
    _log.info(f"Read config from {run_dir}")

    model_ing = import_source(run_dir, "model_ingredient")
    model = model_ing.make_model(**{**config['model'], 'device':device}, _log=_log)
    path = get_model_path(run_dir, epoch)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path, map_location=device))
    model = model.eval()
    _log.info(f"Loaded state dict from {path}")

    if hasattr(model, "odeblock"):
        _log.info(f"Updated times to {[min_end_time, max_end_time]}")
        model.odeblock.min_end_time = min_end_time
        model.odeblock.max_end_time = max_end_time
        model.odeblock.atol = tol
        model.odeblock.rtol = tol


    data_ing = import_source(run_dir, "data_ingredient")
    dset, tl, vl, test_loader = data_ing.make_dataloaders(**{**config['dataset'],
                                                             'device':device},
                                                          _log=_log)
    _log.info("Testing model...")
    test_loss, test_acc = validate(model, test_loader)

    _log.info(f"Test loss = {test_loss:.6f}, Test accuracy = {test_acc:.4f}")
