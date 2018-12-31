"""Adversarial attacks"""
from torch import nn

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
    inp = inp.clone().detach().requires_grad_(True)
    output = model(inp)
    loss = nn.CrossEntropyLoss()(output, label)
    loss.backward()

    perturbation = inp.grad.sign()*epsilon

    return (inp+perturbation).detach()

# # %%
#
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer = nn.Linear(784, 10)
#     def forward(self, input):
#         return self.layer(input.view(input.shape[0],-1))
#
# model = Model()
# input = torch.randn(5,28,28)
# label = torch.LongTensor([5,3,2,1,0])
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
