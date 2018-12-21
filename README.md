# Neural ODE experiments

### Modules
* `modules.py`: Contains the pytorch modules used to build the model to be trained.
* `mnist_dataset.py` A pytorch dataset to handle MNIST images and load them directly on GPU or CPU.
* `training_functions.py`: Contains the `train_on_batch` and `validate` functions used in the training loop. Also has LR scheduling callbacks.
* `data_ingredient.py`: Sacred ingredient which sets up data loaders.
* `model_ingredient.py`: Sacred ingredient which instantiates the required models using the config.

### Scripts
These are sacred experiments.
* `train.py`: Loads the model and data ingredients and uses the `train_on_batch` and `validate` functions to train the model. Also contains the code which creates the optimizer.
* `test.py`: Run inference on trained models by specifying a run directory and epoch to load.
