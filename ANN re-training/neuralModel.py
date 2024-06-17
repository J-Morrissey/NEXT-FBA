import numpy as np
import scipy as sci
from scipy.stats import t
import tensorflow as tf
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score

"""
Class object containing the current value of the hyperparameters
This function is passed to the model builder

Inputs
---------------------------
- nn1: no. of neurons in the first hidden layer
- nn2: no. of neurons in the second hidden layer
- lr: initial learning rate
- L2regularization : 2 Norm regularization factor
- activation: activation function 

Outputs
--------------------------
- object containing the value of the hyperparameters

 Version
-------------------------
v1 - gb 06/06/2022 --> base code
"""


class hyperParams():

    # Initialization method
    def __init__(self, nn1, nn2, lr, L2regularization=0.0001, activation='relu'):
        self.layer1 = nn1
        self.layer2 = nn2
        self.learning_rate = lr
        self.L2regularization = L2regularization
        self.activation = activation

    def __str__(self):
        txt = f"{self.layer1}, {self.layer2}, {self.learning_rate}"
        return txt


"""
Basic function for model building given values of some hyperparameters
Modify this function if different NN is used or if different hyperparameters should be tested

Inputs
---------------------------
- hp: hyperParams object containing the values of the hyperparameters to test

Outputs
--------------------------
- model: tensorflow model object 

 Version
-------------------------
v1 - gb 06/06/2022 --> base code
"""


def build_model(hp):

    # Start building the model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(24,)))

    # Add the first hidden layer NN are tuned
    model.add(
        tf.keras.layers.Dense(
            hp.layer1,
            activation=hp.activation,  # tanh, relu
            kernel_regularizer=tf.keras.regularizers.L2(hp.L2regularization)))

    # Add the second layer, only in NN are different form zero
    if hp.layer2 != 0:
        model.add(
            tf.keras.layers.Dense(
                hp.layer2,
                activation=hp.activation,  # tanh, relu
                kernel_regularizer=tf.keras.regularizers.L2(hp.L2regularization)))

    # Add output layer
    model.add(tf.keras.layers.Dense(1, activation=None,
                                    kernel_regularizer=tf.keras.regularizers.L2(hp.L2regularization)))

    # Tune the Adam learning rate
    hp_lr = hp.learning_rate

    # Compile the model
    # Adam optimizer is selected, loss function is the Mean Squared error and MAE is monitored
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=tf.keras.metrics.MeanAbsoluteError(name='mean_absolute_error'))

    return model


"""
Function to perform hyperparameter tuning. 
Hyperparameter search is performed with Grid search in order to proper split the data by group

Inputs
---------------------------
- X: regressors
- Y: response
- nn1: no. of neurons of the first layer (list with values to test)
- nn2: no. of neurons of the second layer (list with values to test)
- lr: learning rate (list with values to test)
- groups: group index to perform the data splitting
- no_splits: number of Kfold splitting
- max_epochs: number of max training epochs. Early stopping with validation data is implemented

Outputs
--------------------------
- best_hyperp: best hyperparameters values
- mse: list of mse for each test set of hyperparameters
- hyperp: list of tested hyperparameters vaues

 Version
-------------------------
v1 - gb 06/06/2022 --> base code
"""


def hyperparameterOptimization(X, Y, nn1=10, nn2=0, lr=1e-3, groups=None, no_splits=4, max_epochs=20,
                               activation='relu'):
    # Preparing values of the hyperparameters for crossvalidation
    nn1, nn2, lr = np.meshgrid(nn1, nn2, lr)

    # Vector with all possible combinations of hyperparameters
    hyperp = np.concatenate(
        (nn1.reshape(-1, 1), nn2.reshape(-1, 1),
         lr.reshape(-1, 1)), axis=1)

    # Initialization of some lists
    mse = list()

    # Loop over the hyperparameters combinations
    for hyp in hyperp:
        # Storing the hyperparameters in the hyperParams object
        hp = hyperParams(hyp[0], hyp[1], hyp[2], activation=activation)

        # Early stopping function
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=3)

        # Data splitting method
        group_kfold = GroupKFold(n_splits=no_splits)

        # Initialize some lists
        it_mse = []

        for train, test in group_kfold.split(X, Y, groups):
            # Build the NN model with specific hyperparameters, in order to start with a new model every time a
            # different cross-validation set is selected
            model = build_model(hp)

            # Fit the data to the model
            model.fit(X[train], Y[train],
                      batch_size=32,
                      epochs=max_epochs,
                      validation_data=(X[test], Y[test]),
                      callbacks=[stop_early],
                      shuffle=True,
                      verbose=0)

            # Evaluate the model performance
            y_pred = model.predict(X[test])

            # Calculation of the performance on the internal test set
            it_mse.append(mean_absolute_error(Y[test], y_pred))

        # Calculating the average mse for the hyperparameter set and saving it
        mse.append(np.mean(np.array(it_mse)))

    # Find the best hyperparameters
    mse = np.array(mse)
    min_id = np.argmin(mse)
    best_hyperp = hyperp[min_id]

    return best_hyperp, mse, hyperp


"""
Function performing the internal cross-validation of the model. 
This is used to determine the internal performance of the model and to identify the best number of training epochs for 
the final models

Inputs
---------------------------
- X: regressors
- Y: response
- groups: group index to perform the data splitting
- iterations: number of Monte Carlo splitting iterations
- test_fraction: fraction of data tu use as internal validation set
- max_epochs: number of max training epochs. Early stopping with validation data is implemented

Outputs
--------------------------
- mse: MSE for each internal validation iteration
- r2: R2 for each internal validation iteration
- tr_epochs: list of the epochs each internal model was trained for

 Version
-------------------------
v1 - gb 06/06/2022 --> base code
"""


def networkCrossvalidation(X, Y, hp, groups=None, iterations=4, test_fraction=0.2, max_epochs=25):
    # Initialize some lists (general il the Y has more columns)
    if Y.ndim == 1:
        mse = np.ones(1)
        r2 = np.zeros(1)
    else:
        mse = np.ones((1, Y.shape[1]))
        r2 = np.zeros((1, Y.shape[1]))
    Ypred = np.ones((1, ))
    Ycvtest = np.ones((1, ))
    tr_epochs = []

    # Early stopping function
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=3)

    # Monte Carlo datas splitting for groups
    gss = GroupShuffleSplit(n_splits=iterations, test_size=test_fraction)

    for train, test in gss.split(X, Y, groups=groups):
        # Model building function, to start from scratch each time a different cross-validation set is selected
        model = build_model(hp)

        # Fit the data to the model
        history = model.fit(X[train], Y[train],
                            batch_size=32,
                            epochs=max_epochs,
                            validation_data=(X[test], Y[test]),
                            callbacks=[stop_early],
                            shuffle=True,
                            verbose=0)

        # Save the number of training epochs
        tr_epochs.append(len(history.history['loss']))

        # Evaluate the model performance
        y_pred = model.predict(X[test])
        Ypred = np.concatenate((Ypred, y_pred.reshape((-1, ))), axis=0)
        Ycvtest = np.concatenate((Ycvtest, Y[test].reshape((-1, ))), axis=0)

        # Calculate the performance (different for Y with 1 or more columns)
        if Y.ndim == 1:
            mse = np.concatenate((mse, mean_absolute_error(Y[test], y_pred, multioutput='raw_values')), axis=0)
            r2 = np.concatenate((r2, r2_score(Y[test], y_pred, multioutput='raw_values')), axis=0)
        else:
            mse = np.concatenate((mse, mean_absolute_error(Y[test], y_pred, multioutput='raw_values').reshape((1, -1))),
                                 axis=0)
            r2 = np.concatenate((r2, r2_score(Y[test], y_pred, multioutput='raw_values').reshape((1, -1))), axis=0)

    return np.array(mse[1:]), np.array(r2[1:]), np.array(tr_epochs), Ypred[1:], Ycvtest[1:]


"""
This function is used to calculate calculate the Jacobian matrix and all other parameters required for PI prediction

Inputs
---------------------------
- model: model trained 
- x_train: training regressors
- y_train: train response
- regularization: regularization for matrix inversion

Outputs
--------------------------
- SigInv: matrix required in the calculation of the confidence interval
- s2: standard error in calibration

 Version
-------------------------
v1 - gb 06/06/2022 --> base code
"""


def ci_calculation(model, x_train, y_train, regularization=0.01):
    # Calculation of the predictions for the calibration data
    y_calpred = model.predict(x_train).reshape((-1,))

    # Calculation of the Jacobian matrix
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    F = np.zeros((1, trainableParams))

    for xi in x_train:
        xi = xi.reshape((1, len(xi)))
        with tf.GradientTape() as tape:
            y = model(xi, training=False)

        dydx = tape.gradient(y, model.trainable_weights)

        # Unzip of gradient
        gr = np.array([])

        for gi in dydx:
            gi = gi.numpy()

            gr = np.concatenate((gr, gi.reshape((-1,))), axis=0)

        F = np.concatenate((F, gr.reshape((1, -1))), axis=0)

    F = F[1:].copy()

    # Calculation of dof, and standard error
    ff = np.dot(F.transpose(), F)
    finv = sci.linalg.pinv(ff + regularization * np.identity(len(ff)))
    H = np.dot(np.dot(F, finv), F.transpose())
    dof = len(x_train) - np.trace(2 * H - np.dot(H, H))
    s2 = np.sum((y_train - y_calpred) ** 2) / dof

    # Check the norm of the inverted matrix
    if np.linalg.norm(finv) > 2e+4:
        raise Warning('Norm of the inverted matrix exceed 2e+4')

    # Calculation of the big product matrix for CI calculation
    SigInv = np.dot(np.dot(finv, ff), finv)

    return SigInv, s2, dof
