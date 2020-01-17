# Imports
from __future__ import division

# if you have a CUDA-enabled GPU the set the GPU flag as True
GPU=True

import time
from mnist import MNIST
import numpy as np
if GPU:
    import cupy as cp  # You need to have a CUDA-enabled GPU to use this package!
else:
    cp=np

# Parameter setting
thr = [100, 100]  # The threshold of hidden and output neurons
lr = [.2, .2]  # The learning rate of hidden and ouput neurons
lamda = [0.000001, 0.000001]  # The regularization penalty for hidden and ouput neurons
b = [5, 48]  # The upper bound of wight initializations for hidden and ouput neurons
a = [0, 0]  # The lower bound of wight initializations for hidden and ouput neurons
Nepoch = 100  # The maximum number of training epochs
NumOfClasses = 10  # Number of classes
Nlayers = 2  # Number of layers
NhidenNeurons = 400  # Number of hidden neurons
Dropout = [0, 0]
tmax = 256  # Simulatin time
GrayLevels = 255  # Image GrayLevels
gamma = 3  # The gamma parameter in the relative target firing calculation

# General settings
loading = False  # Set it as True if you want to load a pretrained model
LoadFrom = "weights.npy"  # The pretrained model
saving = False  # Set it as True if you want to save the trained model
best_perf = 0
Nnrn = [NhidenNeurons, NumOfClasses]  # Number of neurons at hidden and output layers
cats = [4, 1, 0, 7, 9, 2, 3, 5, 8, 6]  # Reordering the categories

# General variables
images = []  # To keep training images
labels = []  # To keep training labels
images_test = []  # To keep test images
labels_test = []  # To keep test labels
W = []  # To hold the weights of hidden and output layers
firingTime = []  # To hold the firing times of hidden and output layers
Spikes = []  # To hold the spike trains of hidden and output layers
X = []  # To be used in converting firing times to spike trains
target = cp.zeros([NumOfClasses])  # To keep the target firing times of current image
FiringFrequency = []  # to count number of spikes each neuron emits during an epoch

# loading MNIST dataset
mndata = MNIST('MNIST/')
# mndata.gz = False

Images, Labels = mndata.load_training()
Images = np.array(Images)
for i in range(len(Labels)):
    if Labels[i] in cats:
        images.append(np.floor((GrayLevels - Images[i].reshape(28, 28)) * tmax / GrayLevels).astype(int))
        labels.append(cats.index(Labels[i]))

Images, Labels = mndata.load_testing()
Images = np.array(Images)
for i in range(len(Labels)):
    if Labels[i] in cats:
        # images_test.append(TTT[i].reshape(28,28).astype(int))
        images_test.append(np.floor((GrayLevels - Images[i].reshape(28, 28)) * tmax / GrayLevels).astype(int))
        labels_test.append(cats.index(Labels[i]))

del Images, Labels

images = cp.asarray(images)
labels = cp.asarray(labels)
images_test = cp.asarray(images_test)
labels_test = cp.asarray(labels_test)

# Building the model
layerSize = [[images[0].shape[0], images[0].shape[1]], [NhidenNeurons, 1], [NumOfClasses, 1]]
x = cp.mgrid[0:layerSize[0][0], 0:layerSize[0][1]]  # To be used in converting raw image into a spike image
SpikeImage = cp.zeros((layerSize[0][0], layerSize[0][1], tmax + 1))  # To keep spike image

# Initializing the network
np.random.seed(0)
for layer in range(Nlayers):
    W.append(cp.asarray(
        (b[layer] - a[layer]) * np.random.random_sample((Nnrn[layer], layerSize[layer][0], layerSize[layer][1])) + a[
            layer]))
    firingTime.append(cp.asarray(np.zeros(Nnrn[layer])))
    Spikes.append(cp.asarray(np.zeros((layerSize[layer + 1][0], layerSize[layer + 1][1], tmax + 1))))
    X.append(cp.asarray(np.mgrid[0:layerSize[layer + 1][0], 0:layerSize[layer + 1][1]]))
if loading:
    W = np.load(LoadFrom, allow_pickle=True)

SpikeList = [SpikeImage] + Spikes

# Start learning
for epoch in range(Nepoch):
    start_time = time.time()
    correct = cp.zeros(NumOfClasses)
    FiringFrequency = cp.zeros((NhidenNeurons))

    # Start an epoch
    for iteration in range(len(images)):
        # converting input image into spiking image
        SpikeImage[:, :, :] = 0
        SpikeImage[x[0], x[1], images[iteration]] = 1

        # Feedforward path
        for layer in range(Nlayers):
            Voltage = cp.cumsum(cp.tensordot(W[layer], SpikeList[layer]), 1)  # Computing the voltage
            Voltage[:, tmax] = thr[layer] + 1  # Forcing the fake spike
            firingTime[layer] = cp.argmax(Voltage > thr[layer], axis=1).astype(
                float) + 1  # Findign the first threshold crossing
            firingTime[layer][firingTime[layer] > tmax] = tmax  # Forcing the fake spike

            Spikes[layer][:, :, :] = 0
            Spikes[layer][X[layer][0], X[layer][1], firingTime[layer].reshape(Nnrn[layer], 1).astype(
                int)] = 1  # converting firing times to spikes

        FiringFrequency = FiringFrequency + (firingTime[0] < tmax)  # FiringFrequency is used to find dead neurons

        # Computing the relative target firing times
        winner = np.argmin(firingTime[Nlayers - 1])
        minFiring = min(firingTime[layer])
        if minFiring == tmax:
            target[:] = minFiring
            target[labels[iteration]] = minFiring - gamma
            target = target.astype(int)
        else:
            target[:] = firingTime[layer][:]
            toChange = (firingTime[layer] - minFiring) < gamma
            target[toChange] = min(minFiring + gamma, tmax)
            target[labels[iteration]] = minFiring

        # Backward path
        layer = Nlayers - 1  # Output layer

        delta_o = (target - firingTime[layer]) / tmax  # Error in the ouput layer

        # Gradient normalization
        norm = cp.linalg.norm(delta_o)
        if (norm != 0):
            delta_o = delta_o / norm

        if Dropout[layer] > 0:
            firingTime[layer][cp.asarray(np.random.permutation(Nnrn[layer])[:Dropout[layer]])] = tmax

        # Updating hidden-output weights
        hasFired_o = firingTime[layer - 1] < firingTime[layer][:,
                                             cp.newaxis]  # To find which hidden neurons has fired before the ouput neurons
        W[layer][:, :, 0] -= (delta_o[:, cp.newaxis] * hasFired_o * lr[layer])  # Update hidden-ouput weights
        W[layer] -= lr[layer] * lamda[layer] * W[layer]  # Weight regularization

        # Backpropagating error to hidden neurons
        delta_h = (cp.multiply(delta_o[:, cp.newaxis] * hasFired_o, W[layer][:, :, 0])).sum(
            axis=0)  # Backpropagated errors from ouput layer to hidden layer

        layer = Nlayers - 2  # Hidden layer

        # Gradient normalization
        norm = cp.linalg.norm(delta_h)
        if (norm != 0):
            delta_h = delta_h / norm
        # Updating input-hidden weights
        hasFired_h = images[iteration] < firingTime[layer][:, cp.newaxis,
                                         cp.newaxis]  # To find which input neurons has fired before the hidden neurons
        W[layer] -= lr[layer] * delta_h[:, cp.newaxis, cp.newaxis] * hasFired_h  # Update input-hidden weights
        W[layer] -= lr[layer] * lamda[layer] * W[layer]  # Weight regularization

    # Evaluating on test samples
    correct = 0
    for iteration in range(len(images_test)):
        SpikeImage[:, :, :] = 0
        SpikeImage[x[0], x[1], images_test[iteration]] = 1
        for layer in range(Nlayers):
            Voltage = cp.cumsum(cp.tensordot(W[layer], SpikeList[layer]), 1)
            Voltage[:, tmax] = thr[layer] + 1
            firingTime[layer] = cp.argmax(Voltage > thr[layer], axis=1).astype(float) + 1
            firingTime[layer][firingTime[layer] > tmax] = tmax
            Spikes[layer][:, :, :] = 0
            Spikes[layer][X[layer][0], X[layer][1], firingTime[layer].reshape(Nnrn[layer], 1).astype(int)] = 1
        minFiringTime = firingTime[Nlayers - 1].min()
        if minFiringTime == tmax:
            V = np.argmax(Voltage[:, tmax - 3])
            if V == labels_test[iteration]:
                correct += 1
        else:
            if firingTime[layer][labels_test[iteration]] == minFiringTime:
                correct += 1
    testPerf = correct / len(images_test)

    # Evaluating on train samples
    correct = 0
    for iteration in range(len(images)):
        SpikeImage[:, :, :] = 0
        SpikeImage[x[0], x[1], images[iteration]] = 1
        for layer in range(Nlayers):
            Voltage = cp.cumsum(cp.tensordot(W[layer], SpikeList[layer]), 1)
            Voltage[:, tmax] = thr[layer] + 1
            firingTime[layer] = cp.argmax(Voltage > thr[layer], axis=1).astype(float) + 1
            firingTime[layer][firingTime[layer] > tmax] = tmax
            Spikes[layer][:, :, :] = 0
            Spikes[layer][X[layer][0], X[layer][1], firingTime[layer].reshape(Nnrn[layer], 1).astype(int)] = 1
        minFiringTime = firingTime[Nlayers - 1].min()
        if minFiringTime == tmax:
            V = np.argmax(Voltage[:, tmax - 3])
            if V == labels[iteration]:
                correct += 1
        else:
            if firingTime[layer][labels[iteration]] == minFiringTime:
                correct += 1
    trainPerf = correct / len(images)

    print('epoch= ', epoch, 'Perf_train= ', trainPerf, 'Perf_test= ', testPerf)
    print("--- %s seconds ---" % (time.time() - start_time))

    # To save the weights
    if saving:
        np.save("weights", W, allow_pickle=True)
        if testPerf > best_perf:
            np.save("weights_best", W, allow_pickle=True)
            best_perf = val

    # To find and reset dead neurons
    ResetCheck = FiringFrequency < 0.001 * len(images)
    ToReset = [i for i in range(NhidenNeurons) if ResetCheck[i]]
    for i in ToReset:
        W[0][i] = cp.asarray((b[0] - a[0]) * np.random.random_sample((layerSize[0][0], layerSize[0][1])) + a[0])  # r
