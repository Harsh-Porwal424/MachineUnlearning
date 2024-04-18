# Machine Unlearning

## Introduction
This notebook demonstrates the process of unlearning specific classes in a pre-trained image classification model using noise tensors. The unlearning process involves impairing the model's performance on the specified classes and then repairing it to restore performance on the retained classes.

## Requirements
- Python 3.x
- PyTorch
- torchvision
- numpy
- tarfile
- os

## Code Flow
The code can be divided into the following main sections:

1. **Data Loading and Model Initialization:** Downloading and preparing the CIFAR-10 dataset, defining transformations, and initializing the ResNet-18 model.

2. **Training the Model:** Training the model on the entire dataset (all classes).

3. **Unlearning Process:**
   - **Impair Step:** Optimizing noise tensors for classes that need to be unlearned, adding noise to the dataset for those classes, and training the model on the noisy dataset.
   - **Repair Step:** Training the model on the samples from retained classes to restore performance.

4. **Performance Evaluation:** Evaluating the model's performance on the forget and retain classes after each step of the unlearning process.

## Usage
- Ensure the dataset is downloaded and extracted.
- Train the model on the entire dataset.
- Perform the impair and repair steps for unlearning specific classes.
- Evaluate the model's performance after each step.

## Example
```python
# Train the model
history = fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl,
                         grad_clip=grad_clip,
                         weight_decay=weight_decay,
                         opt_func=opt_func)

# Perform unlearning steps
# Impair Step
# Repair Step

# Evaluate the model
history = [evaluate(model, valid_dl)]
