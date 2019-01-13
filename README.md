# Neural ODE and Adversarial Attacks

Code for my blog post on [Neural ODEs and Adversarial Attacks](https://rajatvd.github.io/Neural-ODE-Adversarial/) in which I train neural ODEs on MNIST with different end times and compare their adversarial robustness. I also adversarially train a neural ODE and investigate how this affects its dynamics. It seems like adversarially trained ODEs map different classes of inputs to different equilibria or steady states of the ODE.

Overview of the code:

### Modules
* `modules.py`: Contains the pytorch modules used to build the model to be trained.
  - `ConvODEfunc` is the neural network which parameterizes the dynamics of the neural ODE ( f(z, t; theta) ).
  - `ODEBlock` wraps a general function like `ConvODEfunc` and calls the ODE solver.
  - `ODEBlockRandTime` same as above, but supports random end times.
  - `ODEnetRandTime` is the top level module used for MNIST classification. It has an initial convolution and downsampling layer before passing into the ODE-net, and a final fully connected layer to output class scores.
* `mnist_dataset.py` A pytorch dataset to handle MNIST images and load them directly on GPU or CPU.
* `training_functions.py`: Contains the `train_on_batch` and `validate` functions used in the training loop. Also has LR scheduling callbacks.
* `data_ingredient.py`: Sacred ingredient which sets up data loaders.
* `model_ingredient.py`: Sacred ingredient which instantiates the required ODEnet models using the config.
* `adversarial.py`: Code for adversarial attacks. Includes FGSM and Projected Gradient Descent(PGD).

### Scripts
These are sacred experiments. Use this general command to run them:

`python <script_name>.py with <config_updates>`

Run the following to see the config parameters:

`python <script_name>.py print_config`

* `train.py`: Loads the model and data ingredients and uses the `train_on_batch` and `validate` functions to train the model. Also contains the code which creates the optimizer.
* `test.py`: Run inference on trained models by specifying a run directory and epoch to load.
* `adv_train.py`: Train a model with adversarial data augmentation.
* `adv_test.py`: Test a model with adversarial attacks.
* `time_test.py`: Test a neural ODE model over a range of end times.
