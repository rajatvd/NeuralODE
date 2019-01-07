"""Adversarial attacks"""
from torch import nn
import numpy as np
import torch

# %%

def fgsm(model, inp, label, epsilon=0.3):
    """Performs FGSM attack on model assuming cross entropy loss.

    Parameters
    ----------
    model : nn.Module
        Model to attack
    input : tensor
        Input to perturb adversarially.
    label : tensor
        Target label to minimize score of.
    epsilon : float
        Magnitude of perturbation (the default is 0.3).

    Returns
    -------
    tensor
        Adversarially perturbed input.

    """
    if epsilon == 0:
        return inp.clone().detach()
    inp = inp.clone().detach().requires_grad_(True)
    output = model(inp)
    loss = nn.CrossEntropyLoss()(output, label)
    loss.backward()

    perturbation = inp.grad.sign()*epsilon

    return (inp+perturbation).detach()

# # %%
# import torch
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer = nn.Linear(784, 10)
#     def forward(self, input):
#         return self.layer(input.view(input.shape[0],-1))
#
# device='cuda'
# model = Model().to(device)
# input = torch.randn(5,28,28).to(device)
# inp = input
# label = torch.LongTensor([5,3,2,1,0]).to(device)
#
# # %%
# adv_inp = fgsm(model, input, label)
# with torch.no_grad():
#     print(model(input)[range(len(input)),label])
#     print(model(adv_inp)[range(len(input)),label])


# %%
class AdversarialLoader():
    """Wrap a DataLoader with an adversarial attack.

    Parameters
    ----------
    model : nn.Module
        Model used to generate adversarial examples.
    dataloader : DataLoader
        Data loader to wrap. Assumes it yields batches of inputs and labels.
    attack : function
        Attack function with signature (model, input, label) -> adversarial_input.

    Attributes
    ----------
    model
    dataloader
    attack

    """
    def __init__(self, model, dataloader, attack):
        self.model = model
        self.dataloader = dataloader
        self.attack = attack

    def __iter__(self):
        for input, label in self.dataloader:
            yield self.attack(self.model, input, label), label

    def __len__(self):
        return len(self.dataloader)


# # %%
# import logging
# from data_ingredient import make_dataloaders
# from training_functions import validate
# from tqdm import tqdm
#
# dset, train, val, test = make_dataloaders(32, 0, 0.1, 'cpu', logging.getLogger("dataset"))
#
# adv_test = AdversarialLoader(model, train, fgsm)
# len(adv_test)
#
#
# # %%
# validate(model, test)
#
# # %%
# validate(model, adv_test)

# %%
# inp = input
# epsilon = 0.3
# step_size = 0.01
# num_steps = 40
# random_start = True
# %%
def pgd(model, inp, label,
        epsilon=0.3,
        step_size=0.01,
        num_steps=40,
        random_start=True,
        pixel_range=(-0.5, 0.5)):
    """Short summary.

    Parameters
    ----------
    model : nn.Module
        Model to attack
    inp : tensor
        Input to perturb adversarially.
    label : tensor
        Target label to minimize score of.
    epsilon : float
        Magnitude of perturbation (the default is 0.3).
    step_size : float
        Size of PGD step (the default is 0.01).
    num_steps : float
        Number of PGD steps for one attack. Note that the model is called this
        many times for each attack. (the default is 40).
    random_start : float
        Whether or not to add a uniform random (-epsilon to epsilon) perturbation
        before performing PGD. (the default is True).
    pixel_range : float
        Range to clip the output. (the default is (-0.5, 0.5)).

    Returns
    -------
    tensor
        Adversarially perturbed input.

    """


    adv_inp = inp.clone().detach().cpu().numpy()
    if epsilon == 0:
        return torch.tensor(adv_inp, device=inp.device)

    if random_start:
        adv_inp += np.random.uniform(-epsilon, epsilon, adv_inp.shape)


    for i in range(num_steps):
        inp_var = torch.tensor(adv_inp).to(inp.device).requires_grad_(True)

        output = model(inp_var)
        loss = nn.CrossEntropyLoss()(output, label)
        loss.backward()

        adv_inp += inp_var.grad.sign().cpu().numpy()*step_size

        adv_inp = np.clip(adv_inp, adv_inp-epsilon, adv_inp+epsilon)
        adv_inp = np.clip(adv_inp, *pixel_range)

    return torch.tensor(adv_inp, device=inp.device)

# # %%
# import logging
# from data_ingredient import make_dataloaders
# from training_functions import validate
# from tqdm import tqdm
#
# device = 'cuda'
# model = model.to(device)
#
# dset, train, val, test = make_dataloaders(32, 0, 0.1, device, logging.getLogger("dataset"))
#
# adv_test = AdversarialLoader(model, test, pgd)
# len(adv_test)
#
#
# # %%
# validate(model, test)
#
# # %%
# validate(model, adv_test)
