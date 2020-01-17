# S4NN
The implementation of S4NN presented in "S4NN: temporal backpropagation for spiking neural networks with one spike per neuron" by S. R. Kheradpisheh and T. Masquelier, availbale at: https://arxiv.org/abs/1910.09495.

Two versions of the code are available:
 - The S4NN.py if you want to run the codes with python.
 - The S4NN.ipynb if you want to run the codes on Google CoLab.
  
To run the codes on MNIST dataset, you should first unzip the MNIST.zip file. Then, you should install the python-mnist package. To do so, yo can run the following command:

`$ sudo pip install python-mnist`

If you want to run the codes on GPU, you should set GPU=True. Also, you need to install Cupy package to work with GPU. Cupy is already installed on Google CoLab. You can install it on your own machine by the following command:

`$ sudo pip install cupy`

The pre-trained wight matrix is available at weights_pretrained.npy file. 
