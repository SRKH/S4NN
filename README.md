# S4NN
Implementation of S4NN presented in "S4NN: temporal backpropagation for spiking neural networks with one spike per neuron" by S. R. Kheradpisheh and T. Masquelier that is availbale at: https://arxiv.org/abs/1910.09495.

Two versions of the codes are available:
 - The S4NN.py if you want to run the codes with python.
 - The S4NN.ipynb if you want to run the codes on Google CoLab.
  
In both cases, if you want to run the codes on GPU you, you should set GPU=True. Also, you need to install Cupy package to work with GPU. Cupy is already installed on Google CoLab. You can install it on your own machine by the following command:

$ sudo pip install cupy

You should install the python-mnist package to work with the MNIST dataset. You can install it using the following command:

$ sudo pip install python-mnist
