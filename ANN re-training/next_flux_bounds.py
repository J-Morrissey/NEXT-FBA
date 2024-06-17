from os.path import join
import pickle
import numpy as np
import pandas as pd
from scipy.stats import t
import tensorflow as tf


"""
NEXT-FBA

Main function for bounds prediction

Input
-------------------------
- uptakes: pandas dataframe (experiments names must be index, uptakes names must be column names)
                            NOTE: column names must match the names of the available inputs

- next_flux_dir: main next-flux directory path
- alpha: confidence level for PI calculation. Note that to enlarge PI alpha should approach 1

Output
-------------------------
- res: pandas dataframe of predicted bounds. For each intracellular flux lower bound, median value and upper bound are
       reported


Version
-------------------------
v1 - gb 31/05/2022 --> base code
v1 - gb 17/06/2022 --> update creation of results dataframe

"""


def next_bounds(uptakes, next_flux_dir, alpha=0.95):

    # Load the list of inputs, available outputs and list equilibrium reactions
    with open(join(next_flux_dir, "uptake_input_list.pkl"), "rb") as f:
        uptake_list = pickle.load(f)

    with open(join(next_flux_dir, "available_outputs.pkl"), "rb") as f:
        output_list = pickle.load(f)

    with open(join(next_flux_dir, "equilibrium_reactions.pkl"), "rb") as f:
        equilibrium_reaction = pickle.load(f)

    # Extract order of columns required for NN input and check that all inputs are available
    index_list = []
    for uptk in uptake_list:
        # Check
        if uptk not in uptakes.columns.to_list():
            raise ValueError(f"Required input {uptk} not found")

        # Create list index
        index_list.append(uptakes.columns.to_list().index(uptk))

    # Order pandas columns according to NN inputs
    new_col_index = list()
    for i in index_list:
        new_col_index.append(uptakes.columns.values[i])

    uptakes = uptakes[new_col_index]

    # Convert dataframe to numpy array
    x = uptakes.to_numpy()

    # Initialize prediction containers
    all_pred = np.zeros((len(uptakes), 1))
    res_col_names = list()

    # Take each intracellular flux and load NN and predict
    for i, flx in enumerate(output_list):
        # Load scaler X
        with open(join(next_flux_dir, "utils", f"scalerX_{flx}.pkl"), "rb") as f:
            scalerX = pickle.load(f)

        # Load scaler Y
        with open(join(next_flux_dir, "utils", f"scalerY_{flx}.pkl"), "rb") as f:
            scalerY = pickle.load(f)

        # Load SigInv
        with open(join(next_flux_dir, "utils", f"SigInv_{flx}.pkl"), "rb") as f:
            SigInv = pickle.load(f)

        # Load sigma2
        with open(join(next_flux_dir, "utils", f"sigma2_{flx}.pkl"), "rb") as f:
            sigma2 = pickle.load(f)

        # Load dof
        with open(join(next_flux_dir, "utils", f"dof_{flx}.pkl"), "rb") as f:
            dof = pickle.load(f)

        # Load model
        model = tf.keras.models.load_model(join(next_flux_dir, f"deployed_nn_{flx}"))

        # Scale data
        xs = scalerX.transform(x)

        # Predict bounds
        ys = model.predict(xs).reshape((-1,))

        # Calculate PI
        ci = pi_calculation(model, xs, SigInv, sigma2, dof, alpha=alpha)

        # Calculation of upper and lower bound
        lb = scalerY.inverse_transform(ys - ci)
        ub = scalerY.inverse_transform(ys + ci)

        # Scale predicted bounds
        ypred = scalerY.inverse_transform(ys)

        # Consider the irreversibility of reactions
        if equilibrium_reaction[i] == 0:
            lb[lb < 0] = 0
            ypred[ypred < 0] = 0

        # Save results
        all_pred = np.concatenate((all_pred, lb.reshape((-1, 1)), ypred.reshape((-1, 1)), ub.reshape((-1, 1))), axis=1)
        res_col_names.extend([f"LB_{flx}", f"MEDIAN_{flx}", f"UB_{flx}"])

    res = pd.DataFrame(data=all_pred[:, 1:], index=uptakes.index.values, columns=res_col_names)

    return res


"""
NEXT-FBA
Function for the calculation of the PI. 
This is called by next_bounds.

Input
-------------------------
- model: tesnorflow NN model
- x_test: numpy 2D array of experiment to predict. It is used to calculate the network gradients in the predicted point
- SigInv: calibration gradient information matrix. Required for PI calculation
- sigma2 = calibration standard error. Required for PI calculation
- dof: model degrees of freedom. Required for PI calculation
- alpha: confidence level for PI calculation


Output
-------------------------
- ci: span of the prediction interval


Version
-------------------------
v1 - gb 01/06/2022 --> base code

"""


def pi_calculation(model, x_test, SigInv, sigma2, dof, alpha=0.95):

    # calculation of the gradients for the predicted points
    ci = np.ones((1,))

    for xe in x_test:
        xe = xe.reshape((1, len(xe)))

        with tf.GradientTape() as tape:
            y = model(xe, training=False)

        dydx = tape.gradient(y, model.trainable_weights)

        # Unzip of gradient
        g = np.array([])

        for gii in dydx:
            gii = gii.numpy()

            g = np.concatenate((g, gii.reshape((-1,))), axis=0)

        # Confidence interval calculation
        conf_int = t.isf(1 - alpha, dof) * np.sqrt(sigma2) * np.sqrt(1 + np.dot(np.dot(g, SigInv), g))

        ci = np.concatenate((ci, np.array([conf_int])), axis=0)

    return ci[1:]
