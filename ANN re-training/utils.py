import numpy as np

"""
NEXT-FLUX

Function to determine unique values and elements with that value

Input
-------------------------
- mylist: list for which unique values are searched


Output
-------------------------
 - unique_list: list of unique values
 - groups: index of the unique values for each element of mylist


Version
-------------------------
v1 - gb 31/05/2022 --> base code

"""


def unique(mylist):
    # Initialize a null list
    unique_list = []
    element_idx = []

    # Traverse for all elements
    for x in mylist:
        # Check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)

        # Find the index of the unique element
        element_idx.append(unique_list.index(x))

    # Return the list
    return unique_list, np.array(element_idx, dtype='int32')


"""
NEXT-FLUX

Function providing the root name for C13 samples. 

Input
-------------------------
- sampleID: sample names


Output
-------------------------
 - names: root name of the sample
 - groups: index of the sample groups dividing them according to root name
 

Version
-------------------------
v1 - gb 31/05/2022 --> base code

"""


def groupExtr(sampleID):
    names = list()

    # Print root names of the samples
    for sp in sampleID:
        pos = sp.find('_')
        if pos != -1:
            names.append(sp[0:sp.find('_')])
        else:
            names.append(sp)

    values, groups = unique(names)

    return names, groups


"""
NEXT-FLUX

Delete multiple elements specified by index from the a list

Input
-------------------------
- mylist: list from which elements should be deleted
- indices: numeric indices of elements to delete


Output
-------------------------
- list_object: new list without elements indicated by indices


Version
-------------------------
v1 - gb 31/05/2022 --> base code

"""


def delete_multiple_element(mylist, indices):
    indices = sorted(indices, reverse=True)
    list_object = mylist.copy()
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)
    return list_object


"""
NEXT-FLUX

Autoscaler class for numpy.array with only one dimension. Tries to resemble the behavior of 
sklearn.preprocessing.StandardScaler

Properties
-------------------------
- info: information about the scaler
- mean_: value of the mean of the 1d input vector
- scale_: value of the standard deviation of the 1d input vector


Methods
-------------------------
- fit: calculates mean and std of the given array
- transform: scale the given array based on the mean and std stored in the scaler
- inverse_transform: undo the scaling for the given vector based on the man and std stored in the scaler


Version
-------------------------
v1 - gb 31/05/2022 --> base code

"""


class my1dScaler():

    def __init__(self):
        self.info = "autoscaling of 1d numpy array"
        self.mean_ = 0
        self.scale_ = 1

    def fit(self, x):
        self.mean_ = np.mean(x, axis=0)
        self.scale_ = np.std(x, axis=0)

    def transform(self, x):
        xs = (x - self.mean_)/self.scale_

        return xs

    def inverse_transform(self, x):
        xs = (x * self.scale_) + self.mean_

        return xs


"""
NEXT-FLUX

Calculation of the normalized error as ratio between absolute error and std of training

Input
-------------------------
- y_true: real values
- y_pred: predicted values
- std: the standard deviation for the scaling must be provided
- averaged: True/False calculate the average of all errors


Output
-------------------------
- norm_error: normalized error


Version
-------------------------
v1 - gb 31/05/2022 --> base code

"""

def normalized_error(y_true, y_pred, std=1, averaged=True):
    norm_error = np.absolute(y_true - y_pred) / std

    # averaging
    if averaged:
        norm_error = np.mean(norm_error)

    return norm_error
