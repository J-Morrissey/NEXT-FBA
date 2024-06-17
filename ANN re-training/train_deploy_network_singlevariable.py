import sys
import os
from os.path import join
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error
from FluxSet import FluxSet
from dataAugmentation import smote, noiseDataAugmentation
from neuralModel import hyperparameterOptimization, networkCrossvalidation, hyperParams, build_model, ci_calculation
from utils import groupExtr, my1dScaler

# Some tensorflow settings
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# ----------------------------------------------------------------------------------------------------------------------
#   INPUT SELECTION
# ----------------------------------------------------------------------------------------------------------------------

########################################################################################################################
# Place here the directory to save NEXT-FBA
main_dir = r"NEXT-FBA"

# Selection of the intracellular flux. Select the index of the 
# intracellular variables for which the model will be trained
variable_to_predict = 1

# Selection of parameters for data augmentation
do_smote = True  # True: do smote and noise DA/ False: no data augmentation
sample_to_augment = 3
neighbors_augm = 10
noiseDAsamples = 5
x_noise = 0.03
y_noise = 0.01

# Number of different models to train to identify the one with the minimal loss
n_models = 25

# Load the data from their path.
# All relevant information will be saved into the FluxSet class
neuralFlux = FluxSet(
    r"Extracellular_uptakes_py.xlsx",
    r"Intracellular_fluxes_py.xlsx",
    r"Extracellular_metadata.xlsx",
    r"Intracellular_metadata.xlsx")

# Activation function for each Intracellular flux: it has been selected based on performance (Qcv and Qall)
activation_list = ['tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'relu', 'tanh', 'tanh', 'relu', 'relu', 'relu',
                   'relu', 'relu', 'relu', 'relu', 'tanh', 'relu', 'relu', 'tanh', 'relu', 'tanh', 'tanh', 'tanh',
                   'relu', 'tanh', 'tanh', 'tanh', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'tanh',
                   'relu', 'relu', 'tanh', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu',
                   'relu', 'relu', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh']

# The following vectors have to be modified to perform ANN 
# hyperparameter optimization with gird search.
# Note that times may increase with increasing number of values to test
# These variables have been selected based on preliminary studies
nn1 = np.array([10, 20, 30, 40, 50])  # neurons of the first layer
nn2 = np.array([0, 10, 20, 30, 40])  # neurons of the second layer
lr = np.array([1e-2, 1e-3, 1e-4])  # initial learning rate

# Parameters for ANN training
no_kfold_split = 15  # number of splits for kfold cross-validation
max_epochs = 500  # max number of training epochs
cv_iterations = 100  # nmber of monte carlo cross-validation iterations
mc_test_fraction = 0.05  # fraction of data to place in validation set

########################################################################################################################

# Selection, extraction and printing of the intracellular reaction to predict
flux_list = neuralFlux.intra_metadata_list()
print('The selected intracellular flux is: {}'.format(flux_list[variable_to_predict]))

# Extraction of the data for the selected intracellular flux
X, Y, sampleID = neuralFlux.dataset_extract(flux_list[variable_to_predict])

# Check if the number of sample is sufficient. Threshold is set to have at least 75% of the sample available
if len(X) / len(neuralFlux.Xe) < 0.75:
    print('The selected intracellular flux has not enough samples to perform predictions!')
    quit()  # Quit the program at this point

# Extraction of the root names and group indices for all the available samples
sample_names, group_idx = groupExtr(sampleID)

# ----------------------------------------------------------------------------------------------------------------------
#   DATA AUGMENTATION
# ----------------------------------------------------------------------------------------------------------------------

# Selection of the data augmentation method
if do_smote:
    # Perform data augmentation with SMOTE
    x_train, y_train, names_train, new_idx = smote(X, Y, sample_names, sample_to_augment, neighbors_augm)

    # Data augmentation with random NOISE
    x_train, y_train, names_train = noiseDataAugmentation(x_train, y_train, names_train, noiseDAsamples,
                                                          x_noise=x_noise,
                                                          y_noise=y_noise)
else:
    x_train = X.copy()
    y_train = Y.copy()
    names_train = sample_names.copy()

# ----------------------------------------------------------------------------------------------------------------------
#   INITIAL MODEL ANALYSIS
# ----------------------------------------------------------------------------------------------------------------------

# Assign the groups to new generated calibration data
root_names_train, train_group_idx = groupExtr(names_train)

# Scale of the data - autoscaling is performed to both X and Y
scalerX = StandardScaler()
scalerX.fit(x_train)
scalerY = my1dScaler()
scalerY.fit(y_train)

# Apply scaling. Note y_test is not scaled because it is used only for comparison at the end
x_train_s = scalerX.transform(x_train)
y_train_s = scalerY.transform(y_train)

# Hyperparameters tuning. 
best_hyperp, mse_hyperp, hyperp = hyperparameterOptimization(x_train_s, y_train_s, nn1=nn1, nn2=nn2, lr=lr,
                                                             groups=train_group_idx, 
                                                             no_splits=no_kfold_split, 
                                                             max_epochs=max_epochs,
                                                             activation=activation_list[variable_to_predict])


# Build the class with the value of the best hyperparametrs
opt_hyperp = hyperParams(nn1=best_hyperp[0], nn2=best_hyperp[1], lr=best_hyperp[2],
                         activation=activation_list[variable_to_predict])

print(datetime.now(), ': Optimal hyperparameters found  ', flux_list[variable_to_predict])
print('The best hyperparameters are: ', opt_hyperp)

# NN model cross-validation to identify number of training epochs
# Evaluation of the internal performances, definition of the number of training epochs
mse_cv, q2_cv, train_epochs, ycvpred, ycvtest = networkCrossvalidation(x_train_s, y_train_s, opt_hyperp,
                                                                       groups=train_group_idx, 
                                                                       iterations=cv_iterations,
                                                                       test_fraction=mc_test_fraction, 
                                                                       max_epochs=max_epochs)

print(datetime.now(), ': Model cross-validated ', flux_list[variable_to_predict])


# ----------------------------------------------------------------------------------------------------------------------
#   MODEL TRAINING
# ----------------------------------------------------------------------------------------------------------------------

# Train several models and take the one with the minimum loss
model = list()
tr_error = list()

for nm in range(n_models):
    mdl = build_model(opt_hyperp)

    # Fit the data to the model
    history = mdl.fit(x_train_s, y_train_s,
                      batch_size=32,
                      epochs=int(np.round(np.mean(train_epochs))),
                      validation_data=None,
                      shuffle=True,
                      verbose=0)

    model.append(mdl)
    tr_error.append(history.history['mean_absolute_error'][-1])

# Find the mest model
tr_error = np.array(tr_error)
model = model[np.argmin(tr_error)]

print(datetime.now(), ': Model trained ', flux_list[variable_to_predict])

# ----------------------------------------------------------------------------------------------------------------------
#   CONFIDENCE INTERVAL CALCULATION
# ----------------------------------------------------------------------------------------------------------------------

# Calculation of the elements required for PI calculation
SigInv, sigma2, dof = ci_calculation(model, x_train, y_train, regularization=0.05)

print(datetime.now(), ': CI calculated ', flux_list[variable_to_predict])

# ----------------------------------------------------------------------------------------------------------------------
#   SAVING STUFF
# ----------------------------------------------------------------------------------------------------------------------


if not os.path.exists(main_dir):
    os.mkdir(main_dir)

# Saving model
tf.keras.models.save_model(model, join(main_dir, f"deployed_nn_{flux_list[variable_to_predict]}"))

# Save scalerX
with open(join(main_dir, "utils", f"scalerX_{flux_list[variable_to_predict]}.pkl"), 'wb') as f:
    pickle.dump(scalerX, f)

# Save scalerY
with open(join(main_dir, "utils", f"scalerY_{flux_list[variable_to_predict]}.pkl"), 'wb') as f:
    pickle.dump(scalerY, f)

# Save SigInv
with open(join(main_dir, "utils", f"SigInv_{flux_list[variable_to_predict]}.pkl"), 'wb') as f:
    pickle.dump(SigInv, f)

# Save Sigma2
with open(join(main_dir, "utils", f"sigma2_{flux_list[variable_to_predict]}.pkl"), 'wb') as f:
    pickle.dump(sigma2, f)

# Save dof
with open(join(main_dir, "utils", f"dof_{flux_list[variable_to_predict]}.pkl"), 'wb') as f:
    pickle.dump(dof, f)

# Save cross-validation performance
save_list = [q2_cv, mse_cv]
with open(join(main_dir, "utils", f"res_crossval_{flux_list[variable_to_predict]}.pkl"), 'wb') as f:
    pickle.dump(save_list, f)
