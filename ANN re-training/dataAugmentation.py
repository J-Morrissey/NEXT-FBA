import numpy as np
from sklearn.utils import shuffle
import random
from sklearn.neighbors import NearestNeighbors
import re

"""
Basic function for the calculation of the composition between two points

Inputs
---------------------------
- xi: base point
- xn: neighbor
- gap: fraction of the distance from base point

 Outputs
 --------------------------
 - x: new point
 
 Version
-------------------------
v1 - gb 06/06/2022 --> base code
"""


def pointGen(xi, xn, gap):
    dist = xn - xi
    x = xi + gap * dist
    return x


"""
SMOTE engine 

Inputs
---------------------------
- X: regressors
- Y: response
- idx: index of the base sample
- nn_idx: index of the nearest neighbors
- percDA: percentage of data to augment
- nn_idx: index of the nuw samples
- name: root sample name to keep track of the generator sample

 Outputs
 -------------------------- 
 - Xnew: new regressors
 - Ynew: new response
 - names: names of the new samples
 - new_idx: index of the new samples
 
 Version
-------------------------
v1 - gb 06/06/2022 --> base code
"""


def smote_engine(X, Y, idx, nn_idx, percDA, new_idx, name):
    Xnew = np.ones((1, np.shape(X)[1]))
    if Y.ndim == 1:
        Ynew = np.ones((1,))
    else:
        Ynew = np.ones((1, np.shape(Y)[1]))
    names = list()

    sp_idx = 1
    while percDA != 0:
        # updating the new idx
        new_idx += 1

        # selection of the nn for generation
        gen_idx = random.sample(list(nn_idx), 1)

        # gap
        gap = random.random()

        # generation of the xnew and ynew
        xn = pointGen(X[idx], X[gen_idx[0]], gap)
        yn = pointGen(Y[idx], Y[gen_idx], gap)

        # adding new samples
        Xnew = np.concatenate((Xnew, xn.reshape((1, -1))), axis=0)
        Ynew = np.concatenate((Ynew, yn), axis=0)
        names.append(f"{name}_DA{str(sp_idx)}_MEDIAN")

        # update sample index and percentage of DA samples
        sp_idx += 1
        percDA -= 1

    return Xnew[1:], Ynew[1:], names, new_idx


"""
Perform SMOTE data augmentation on the data

Inputs
---------------------------
- X: regressors
- Y: response
- sampleID: sample names
- percDA: percentage of sample to augment [0, inf)
- k: number of neighbors

 Outputs
 -------------------------- 
 - Xnew: new regressors
 - Ynew: new response
 - names: names of the new samples
 - new_idx: index of the new samples 
 
 Version
-------------------------
v1 - gb 06/06/2022 --> base code
"""


def smote(X, Y, sampleID, percDA, k):
    # initialization
    names = list()
    new_idx = 0

    for sp in sampleID:
        names.append(sp[0:sp.find('_')])

    # selection of the samples to augment
    if percDA < 1:
        Xs, Ys, spNames = shuffle(X, Y, names)
        to_here = round(len(Xs) * percDA)
        Xs = Xs[0:to_here]
        Ys = Ys[0:to_here]
        spNames = spNames[0:to_here]
        percDA = 1
    else:
        Xs = X
        Ys = Y
        spNames = names

    percDA = round(percDA)

    # creating the initial matrices
    Xnew = X.copy()
    Ynew = Y.copy()
    newNames = sampleID.copy()

    # loop over the samples
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='brute').fit(Xs)
    distances, indices = nbrs.kneighbors(Xs)
    for i in range(len(Xs)):
        nn_idx = np.delete(indices[i], 0)

        # call the SMOTE engine
        xnew, ynew, new_names, new_idx = smote_engine(Xs, Ys, i, nn_idx, percDA, new_idx, spNames[i])

        # adding new sample to the entire dataset
        Xnew = np.concatenate((Xnew, xnew), axis=0)
        Ynew = np.concatenate((Ynew, ynew), axis=0)
        for ni in new_names:
            newNames.append(ni)

    return Xnew, Ynew, newNames, new_idx


"""
Perform NOISE data augmentation on the data

Inputs
---------------------------
- X: regressors
- Y: response
- sampleID: sample names
- numSample: number of sample to generate for each original one
- x_noise: fraction of noise to add on X (0.05)
- y_noise: fraction of noise to add on Y (0.0)

 Outputs
 -------------------------- 
 - Xnew: new regressors
 - Ynew: new response
 - names: names of the new samples
 
 Version
-------------------------
v1 - gb 06/06/2022 --> base code
"""


def noiseDataAugmentation(X, Y, sampleID, numSample, x_noise=0.05, y_noise=0.0):
    # initialization
    names = list()

    for sp in sampleID:
        names.append(sp[0:sp.find('_')])

    # preparation of the data to fill
    Xnew = X.copy()
    Ynew = Y.copy()
    newNames = sampleID.copy()

    for i, xp in enumerate(X):
        yp = Y[i]

        # performing noise augmentation
        xnew = xp + xp * x_noise * np.random.normal(0, 1, (numSample, X.shape[1]))
        if Y.ndim == 1:
            ynew = yp + yp * y_noise * np.random.normal(0, 1, (numSample,))
        else:
            ynew = yp + yp * y_noise * np.random.normal(0, 1, (numSample, Y.shape[1]))
        # print(ynew)

        # adding new samples to original ones
        Xnew = np.concatenate((Xnew, xnew), axis=0)
        Ynew = np.concatenate((Ynew, ynew), axis=0)

        # new names of the samples
        for j in range(numSample):
            newNames.append(f"{names[i]}_noiseDA{str(j + 1)}_MEDIAN")

    return Xnew, Ynew, newNames
