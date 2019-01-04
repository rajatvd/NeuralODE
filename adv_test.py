"""Test a ODEnet on the MNIST dataset with adversarial examples.
"""
# %%
from functools import partial

import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver
from pytorch_utils.sacred_utils import get_model_path, read_config, import_source

from training_functions import validate
import adversarial as adv

# %%
ATTACKS = {
    'fgsm':adv.fgsm,
    'pgd':adv.pgd
}

# %%
ex = Experiment('adv_test_mnist')
SAVE_DIR = "runs/AdvTestMnist"
ex.observers.append(FileStorageObserver.create(SAVE_DIR))

@ex.config
def input_config():
    """Parameters for sampling using the given model"""
    run_dir = 'runs/ODEMnistClassification/17'
    epoch = 'latest'
    device = 'cpu'
    epsilon = 0.3 # epsilon for attack
    attack = 'fgsm' # type of attack, currently: [fgsm, pgd]
    min_end_time = 10
    max_end_time = 100
    tol = 1e-3

    pgd_step_size = 0.01
    pgd_num_steps = 40
    pgd_random_start = True

@ex.automain
def main(run_dir,
         epoch,
         device,
         attack,
         epsilon,
         min_end_time,
         max_end_time,
         tol,
         pgd_step_size,
         pgd_num_steps,
         pgd_random_start,
         _log):

    config = read_config(run_dir)
    _log.info(f"Read config from {run_dir}")

    model_ing = import_source(run_dir, "model_ingredient")
    model = model_ing.make_model(**{**config['model'], 'device':device}, _log=_log)
    path = get_model_path(run_dir, epoch)
    model.load_state_dict(torch.load(path))
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

    if attack == 'pgd':
        attack_fn = partial(ATTACKS[attack],
                            epsilon=epsilon,
                            step_size=pgd_step_size,
                            num_steps=pgd_num_steps,
                            random_start=pgd_random_start)
    else:
        attack_fn = partial(ATTACKS[attack], epsilon=epsilon)

    adv_test_loader = adv.AdversarialLoader(model, test_loader, attack_fn)

    _log.info("Testing model...")
    test_loss, test_acc = validate(model, adv_test_loader)

    _log.info(f"Test loss = {test_loss:.6f}, Test accuracy = {test_acc:.4f}")
