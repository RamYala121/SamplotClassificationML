# Simple CNN Classifier
This is a very simple repo for explaining basic concepts about 
Convolutional Neural Networks (CNNs) to beginners.
The example exploits the PyTorch library (https://pytorch.org/) 
for performing a basic binary classification task on the Kaggle
Dogs vs. Cats dataset (https://www.kaggle.com/c/dogs-vs-cats/data).

## Preparing the dataset
1) Put your raw JPEG samplot files in the repo-local ``train`` folder:
   ``SimpleCNNClassifier/train``
2) Run ``prepare_dataset.py`` to split the files into ``positive`` and ``negative`` subfolders.

If you also have validation images, put them in ``SimpleCNNClassifier/val`` with the same class structure:
- ``SimpleCNNClassifier/val/positive``
- ``SimpleCNNClassifier/val/negative``

## Training
Use the script ``train.py`` in order to train a CNN for this purpose.

## Validation
Use the script ``eval.py`` in order to evaluate the CNN performance on this task.