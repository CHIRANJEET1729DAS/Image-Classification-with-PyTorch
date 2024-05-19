# Image-Classification-with-PyTorch
This project demonstrates a simple image classification task using PyTorch, a popular open-source machine learning library for Python. The goal is to train a convolutional neural network (CNN) to classify images from the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes (e.g., airplanes, cars, birds, etc.).

## Getting Started

To run this project locally, you'll need to have Python 3 and PyTorch installed on your system. You can install PyTorch by following the official installation guide: https://pytorch.org/get-started/locally/

Once you have PyTorch installed, you can clone this repository:

git clone https://github.com/aayushsharma-io/Image-Classification-with-PyTorch.git
cd Image-Classification-with-PyTorch

## In case you don't have a GPU or minimum hardware requirements:
# Google Colab
https://colab.research.google.com/drive/1gInQmLdXsVSMp6QFnJzkrpcKpTlmC9FI?usp=sharing


## Dataset

The CIFAR-10 dataset will be automatically downloaded and extracted to the `./data` directory when you run the code. This dataset is widely used for benchmarking machine learning algorithms and consists of:

- 50,000 training images
- 10,000 test images
- 10 classes (airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks)

## Project Structure

The project consists of a single Python file, `main.py`, which contains the following components:

1. **Neural Network Architecture**: A simple CNN architecture is defined using PyTorch's `nn.Module` class. The network consists of two convolutional layers, two max-pooling layers, and three fully connected layers.

2. **Data Preprocessing**: The CIFAR-10 dataset is loaded and preprocessed using PyTorch's `torchvision` module. The images are normalized and converted to PyTorch tensors.

3. **Training Loop**: The neural network is trained for two epochs using stochastic gradient descent (SGD) optimizer and cross-entropy loss function. During training, the loss is printed every 2000 mini-batches to monitor the progress.

## Running the Project

To run the project, simply execute the `main.py` file:

python3 main.py

This will start the training process, and you should see the loss values being printed to the console as the network learns to classify the images.

## FUN
I programme or code projects for fun :)
follow me for more projects like this.
