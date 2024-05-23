from os.path import join, isdir
import pickle
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from next_pca import pca_pred
from next_flux_bounds import next_bounds

"""
NEXT-FLUX

Main function of the package

Input
-------------------------
- data_to_predict: pandas.DataFrame with uptakes (experiments on rows, uptakes on columns). Row and column index must be
                    provided
                    
- main_path: path of the directory containing NEXT-FLUX
- alpha: confidence level of NN predicted intracellular fluxes. It is used to determine LB and UB
- exclude_different: Flag to avoid prediction for experiments with differences from the training ones (True: predict 
                        only similar/ False: predict all)
                        
- condition: rationale to define an experiment different from the training one. It is based on PCA projection 
                -- 'T2': experiments outside Hotelling's T2 limit are excluded
                -- 'SPE': experiments outside SPE limit are excluded
                -- 'all': only experiments inside both limits are included
                -- 'both' experiments outside both limits are excluded
                
- condition_alpha: select the confidence level of PCA diagnostics ('95'/'99')
- shsh: stop printing of info (True/False)
- save_bounds: flag for saving the predicted bounds as Excel file in the current directory (True/False)


Output
-------------------------
- similarity: pandas DataFrame containing the similarity factors for new experiments


Version
-------------------------
v1 - gb 17/06/2022 --> initial step (data organization, data imputation, pca) and bounds prediction
v1 - gb 23/06/2022 --> fixing organization of the input DataFrame and return/save similarity factors

"""


def next_flux(data_to_predict, main_path, alpha=0.95, exclude_different=False, condition='both', condition_alpha='95',
              shsh=False, save_bounds=False, save_similarity=False):
    # Check the data are Dataframe type
    if not isinstance(data_to_predict, pd.DataFrame):
        raise TypeError(f"Wring data type. Uptakes data must be pandas.DataFrame")

    # Check main_path is a path
    if not isdir(main_path):
        raise ValueError(f"Invalid network path. Please provide a valid path for the network directory")

    # Identification of the available uptakes
    with open(join(main_path, "uptake_input_list.pkl"), "rb") as f:
        uptake_list = pickle.load(f)

    missing_uptakes = list()
    for upt in uptake_list:
        if upt not in data_to_predict.columns.tolist():
            missing_uptakes.append(upt)

    # Count the number of missing data (NaNs) in the given data
    n_miss = sum(data_to_predict.isna().sum().values)

    if not shsh:
        print(f"The number of missing values in the given uptakes is: {n_miss}")

    # Missing data imputation. Note that at this point data has all uptakes ordered as required
    if n_miss != 0 or len(missing_uptakes) != 0:
        data = missing_data_imputation(data_to_predict, main_path, uptake_list)
    else:
        # Order the columns of the dataframe
        data = data_to_predict[uptake_list]

    # PCA for elimination of the samples outside confidence limits
    with open(join(main_path, "C13_pca_trained_model.pkl"), 'rb') as fl:
        pcamdl = pickle.load(fl)

    pred_pca_mdl = pca_pred(pcamdl, data)

    # Assessment of experiment similarity
    is_similar, similarity_fact = check_similarity(pcamdl, pred_pca_mdl, condition=condition, alpha=condition_alpha,
                                                   shsh=shsh)
    exp_names = data.index.tolist()
    is_diff = list()
    if not shsh:
        print('The following experiments are different from the training ones:')

    for i, rs in enumerate(is_similar):
        if not rs:
            is_diff.append(exp_names[i])
            if not shsh:
                print(f"\t - {exp_names[i]}")

    if not shsh:
        if len(is_diff) == 0:
            print("\t no experiment is different")

    # Save similarity factors
    similarity = pd.DataFrame({'Distance form average': similarity_fact[:, 0],
                               'Difference in correlation': similarity_fact[:, 1]},
                              index=data.index.tolist())

    if save_similarity:
        date = datetime.now()
        similarity.to_excel(f"similarity_{date.day}_{date.month}_{date.year}_{date.hour}_{date.minute}.xlsx")

    # Exclusion of the different samples?
    if exclude_different:
        data = data.drop(is_diff, axis=0)
        if not shsh:
            print(f"Experiments excluded from bounds prediction: {len(list(is_diff))}")
    elif not exclude_different and len(is_diff) > 0:
        warnings.warn("DIFFERENT EXPERIMENTS FOUND. Prediction of different experiments is still performed. "
                      "\nProceed with caution, predictions may not be accurate!")

    # Predict bounds
    bounds_df = next_bounds(data, main_path, alpha=alpha)

    if save_bounds:
        date = datetime.now()
        bounds_df.to_excel(f"predicted_bounds_{date.day}_{date.month}_{date.year}_{date.hour}_{date.minute}.xlsx")

    print(bounds_df)  # This print can be deleted in the final version. It is here only to check predictions

    return similarity


"""
NEXT-FLUX

Function for missing uptakes imputation. The method is taken from Garcia Munoz et al., 2004, 

Input
-------------------------
- data: panda.DataFrame containing the uptakes with missing values and missing columns
- main_path: path of the directory containing NEXT-FLUX
- uptake_list: list of the ordered required input of the NN
- toll: tolerance for data imputation
- max_its: max number of iterations in imputation cycle


Output
-------------------------
- new_data: pandas.DataFrame of uptakes with filled missing data and columns


Version
-------------------------
v1 - gb 17/06/2022 --> base code
v1 - gb 23/06/2022 --> fixing organization of the input DataFrame

"""


def missing_data_imputation(data, main_path, uptake_list, toll=1e-14, max_its=1000):
    # Load PCA model
    with open(join(main_path, "C13_pca_trained_model.pkl"), 'rb') as fl:
        pcamdl = pickle.load(fl)

    # Extract scaler for data scaling
    scaler = pcamdl['scaler']

    # Create an empty dataframe with columns ordered in the desired way
    new_data = pd.DataFrame(columns=uptake_list)

    # Place data in the empty database
    for upt in new_data.columns.tolist():
        try:
            new_data[upt] = data[upt].copy()
        except:
            pass

    # Data replacement (done for each experiment)
    for i in range(len(new_data)):
        # Data extraction
        x = new_data.iloc[i, :]

        # Loadings extraction
        isnan = x.isna().values
        Pi = pcamdl['P']
        pstar = pcamdl['P'][isnan == False, :]
        pcanc = pcamdl['P'][isnan == True, :]

        # Scaling of x
        xs = scaler.transform(x.values.reshape((1, -1)))
        xs = xs.reshape((-1,))

        # x extraction
        xstar = xs[isnan == False]

        # Calculation of the scores and predicted x
        tau = np.dot(pstar.transpose(), xstar)
        k = 0

        while True:
            xhats = np.dot(tau, pcanc.transpose())
            xi = xs.copy()
            xi[isnan == True] = xhats
            tau_n = np.dot(xi, Pi)
            err = np.linalg.norm(tau - tau_n)

            # Updates
            tau = tau_n.copy()
            k = k + 1

            if err < toll or k > max_its:
                break

        xhats = np.dot(tau, pcanc.transpose())

        # Perform the unscaling
        xs[isnan == True] = xhats
        xs = scaler.inverse_transform(xs.reshape((1, -1)))
        xs = xs.reshape((-1,))

        # Place the predicted x (xhat) into the dataframe
        xhat = xs[isnan == True]
        new_data.iloc[i, isnan == True] = xhat

        new_data.to_excel('missing data imputation.xlsx')
    return new_data


"""
NEXT-FLUX

Function for the identification of the samples dissimilar to calibration ones

Input
-------------------------
- pcamdl: PCA model done on the training experiments
- pred_pca_mdl: PCA projection of the new experiments
- condition: rationale to define an experiment different from the training one. It is based on PCA preojection 
                -- 'T2': experiments outside Hotelling's T2 limit are excluded
                -- 'SPE': experiments outside SPE limit are excluded
                -- 'all': only experiments inside both limits are included
                -- 'both' experiments outside both limits are excluded
                
- alpha: select the confidence level of PCA diagnostics ('95'/'99')


Output
-------------------------
- is_similar: list with sample similar to the calibration ones. True: sample is similar - False: sample is not similar


Version
-------------------------
v1 - gb 17/06/2022 --> base code
v1 - gb 23/06/2022 --> return similarity factors

"""


def check_similarity(pcamdl, pred_pca_mdl, condition='both', alpha='95', shsh=False):
    # Extraction of the CI according to alpha
    if alpha == '95':
        T2lim = pcamdl['T2_lim']
        SPElim = pcamdl['SPE_lim']
    elif alpha == '99':
        T2lim = pcamdl['T2_lim_99']
        SPElim = pcamdl['SPE_lim_99']
    else:
        raise ValueError(f"The given alpha value {alpha} is not supported. Accepted values are 95 or 99")

    # Print some info
    if not shsh:
        print("Testing experiment similarity - recommended cutoff value = 1")
        print(f"\t Method: {condition}")

    # Calculation of the similarity factor to return
    similarity_fact = np.array([pred_pca_mdl['T2new'] / T2lim, pred_pca_mdl['SPEnew'] / SPElim]).transpose()

    # Evaluate the condition for each sample
    is_similar = list()
    for i in range(len(pred_pca_mdl['T2new'])):
        if condition == 'T2':
            if pred_pca_mdl['T2new'][i] <= T2lim:
                is_similar.append(True)
            else:
                is_similar.append(False)

        elif condition == 'SPE':
            if pred_pca_mdl['SPEnew'][i] <= SPElim:
                is_similar.append(True)
            else:
                is_similar.append(False)

        elif condition == 'all':
            if pred_pca_mdl['T2new'][i] <= T2lim and pred_pca_mdl['SPEnew'][i] <= SPElim:
                is_similar.append(True)
            else:
                is_similar.append(False)

        elif condition == 'both':
            if pred_pca_mdl['T2new'][i] > T2lim and pred_pca_mdl['SPEnew'][i] > SPElim:
                is_similar.append(False)
            else:
                is_similar.append(True)

        # Print some info
        if not shsh:
            print(f"\t - {pred_pca_mdl['sample_names'][i]}: distance from average="
                  f"{round(pred_pca_mdl['T2new'][i] / T2lim, 2)} "
                  f"| difference in correlation={round(pred_pca_mdl['SPEnew'][i] / SPElim, 2)}")

    return is_similar, similarity_fact