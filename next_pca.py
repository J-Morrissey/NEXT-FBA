from os.path import join
import numpy as np
import pandas as pd
import pickle
# import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from scipy.stats import f, chi2
import matplotlib.pyplot as plt


"""
Function to apply pca trained non C13 uptakes on new data
NOTE: this function does not check correspondence between variables. It supposes that the oreder of columns in given x 
equal to the one of the data (is given as output)

Input
----------------
- xnew: pandas dataframe (index: sample names, columns header: variable names)
- pca_path: path of the pca model file ("C13_pca_trained_model.pkl")
- plot_scores: plot scores of the components in the list. 2 components must be provided. If False no plot
- plot_diagnostics: plot outlier map. True / False
- plot_weights: plot model weights of the component. Must be int. if False no plot

Output
-----------------------------
- None
                    
Version
----------------------------
v1 - gb 31/05/2022

"""


def apply_pca(xnew, pca_path='', plot_scores=[1, 2], plot_diagnostics=True, plot_weights=1):

    # Load trained pca
    with open(join(pca_path, "C13_pca_trained_model.pkl"), 'rb') as fl:
        pcamdl = pickle.load(fl)

    # Project new data into the pca space
    predmdl = pca_pred(pcamdl, xnew)

    # Plot scores
    if not isinstance(plot_scores, bool):
        # scores plot
        plt.figure(figsize=(10, 17))
        plt.scatter(pcamdl['T'][:, plot_scores[0]-1], pcamdl['T'][:, plot_scores[1]-1], s=50,
                    marker="s", color='gray')  # C13 data
        plt.scatter(predmdl['Tnew'][:, plot_scores[0]-1], predmdl['Tnew'][:, plot_scores[1]-1], s=40)  # new data

        # Plot origin lines
        plt.plot([plt.xlim()[0], plt.xlim()[1]], [0, 0], color='black', linewidth=0.9)
        plt.plot([0, 0], [plt.ylim()[0], plt.ylim()[1]], color='black', linewidth=0.9)

        plt.xlabel(f"scores component {plot_scores[0]} (R2x={np.round(100*pcamdl['expl_var'][plot_scores[0]-1], 1)} %)")
        plt.ylabel(f"scores component {plot_scores[1]} (R2x={np.round(100*pcamdl['expl_var'][plot_scores[1]-1], 1)} %)")

        plt.legend(['C13 data', 'new data'])

        plt.show()

    # Plot diagnostics
    if plot_diagnostics:
        # outlier map
        plt.figure(figsize=(10, 17))
        plt.scatter(pcamdl['T2'], pcamdl['SPE'], s=50, marker="s", color='gray')  # C13 data
        plt.scatter(predmdl['T2new'], predmdl['SPEnew'], s=40)  # new data

        # Plot origin lines
        plt.plot([pcamdl['T2_lim'], pcamdl['T2_lim']], [0, plt.ylim()[1]], color='black', linewidth=0.9)
        plt.plot([0, plt.xlim()[1]], [pcamdl['SPE_lim'], pcamdl['SPE_lim']], color='black', linewidth=0.9)

        plt.xlim([0, plt.xlim()[1]])
        plt.ylim([0, plt.ylim()[1]])

        plt.xlabel(f"Hotelling's T2 (R2x={np.round(100*np.sum(pcamdl['expl_var']), 1)} %)")
        plt.ylabel(f"SPE ")

        plt.legend(['C13 data', 'new data', '95% confidence limits'])

        plt.show()

    # Plot loadings
    if not isinstance(plot_weights, bool):
        plt.figure(figsize=(10, 17))
        plt.bar(np.arange(xnew.shape[1]), pcamdl['P'][:, plot_weights-1])

        plt.xlabel('uptakes')
        plt.ylabel(f"weights of component {plot_weights}")
        plt.xticks(np.arange(xnew.shape[1]), labels=xnew.columns.tolist(), rotation=90)

        plt.show()

    # Prepare things to return


"""
Function to perform PCA

Input
-------------------------------------
- X: pandas dataframe (index:sample names/number, columns: variable names)
- ncomp: number of components
- cross_val: if True perform cross-validation
- cross_val_split: number of splits for K fold cross-validation
- sample_groups: list with groups of samples. It performs grouped cv. If samples have no groups provide a list with the
                    index of samples
- shsh: if true function does not plot results
- alpha: confidence level for CI calculation

Output
-----------------------------
- dictionary with: scores (T), weights (P), SPE, Hotelling's T2, confidence limits for T2 and SPE, explained variance, 
                    scaler, pca model
                    
Version
----------------------------
v1 - gb 31/05/2022

"""


def pca(X, ncomp, cross_val=False, cross_val_split=6, sample_groups=[], shsh=False, alpha=0.95):

    # Work on the input
    x = X.to_numpy()

    # Autoscale data
    scaler = StandardScaler()
    scaler.fit(x)
    xs = scaler.transform(x)

    # Perform cross-validation of PCA model
    if cross_val:
        # Initialize some variables
        rmsecv = np.zeros((ncomp, cross_val_split))
        rmsec = np.zeros((ncomp, cross_val_split))
        r2 = np.zeros((ncomp, cross_val_split))
        k = 0
        crossval = GroupKFold(n_splits=cross_val_split)

        for train_index, test_index in crossval.split(xs, groups=sample_groups):
            xtrain = xs[train_index]
            xtest = xs[test_index]

            for ncp in range(ncomp):
                # Train model
                mdl = PCA(n_components=ncp+1)
                mdl.fit(xtrain)

                # Train prediction
                xtp = mdl.transform(xtrain)
                xtr = mdl.inverse_transform(xtp)

                # Test model
                xp= mdl.transform(xtest)

                # Reconstruct x
                xr = mdl.inverse_transform(xp)

                # Calculation of the rmsecv
                rmsecv[ncp, k] = mean_squared_error(xtest, xr, squared=False)
                rmsec[ncp, k] = mean_squared_error(xtrain, xtr, squared=False)
                r2[ncp, k] = np.sum(mdl.explained_variance_ratio_)

            k = k + 1

        # Print cross-val results
        stats = pd.DataFrame({'component': np.arange(1, ncomp+1), 'R2x': np.mean(r2, axis=1),
                              'RMSEC': np.mean(rmsec, axis=1), 'RMSECV': np.mean(rmsecv, axis=1)})
        if not shsh:
            print('Cross-validation results')
            print(stats)

    # Build pca
    mdl = PCA(n_components=ncomp)
    mdl.fit(xs)

    # Decompose given data
    t = mdl.transform(xs)
    p = mdl.components_.transpose()

    # Print cross-val results
    stats = pd.DataFrame({'component': np.arange(1, ncomp+1), 'eigenvalues': np.var(t, axis=0),
                          'R2x': mdl.explained_variance_ratio_, 'R2x tot': np.cumsum(mdl.explained_variance_ratio_)})

    if not shsh:
        print(stats)

    # Calculation of SPE
    xr = mdl.inverse_transform(t)
    E = xs - xr
    spe = np.sum(E * E, axis=1)

    # Calculation of T2
    lambdat = np.var(t, axis=0) ** (-1)
    tsqs = np.sum(np.dot(t * t, np.diag(lambdat)), axis=1)

    # Calculation of the CI (Wise & Gallager 1996)
    tsqlim = ncomp * (len(x) - 1) / (len(x) - ncomp) * f.isf(1 - alpha, ncomp, len(x) - ncomp)
    tsqlim99 = ncomp * (len(x) - 1) / (len(x) - ncomp) * f.isf(1 - 0.99, ncomp, len(x) - ncomp)

    spelim = np.var(spe) / (2 * np.mean(spe)) * chi2.isf(1 - alpha, (2 * np.mean(spe ** 2) / np.var(spe)))
    spelim99 = np.var(spe) / (2 * np.mean(spe)) * chi2.isf(1 - 0.99, (2 * np.mean(spe ** 2) / np.var(spe)))

    dict_to_return = {
        'components': ncomp,
        'T': t,
        'P': p,
        'T2': tsqs,
        'SPE': spe,
        'T2_lim': tsqlim,
        'T2_lim_99': tsqlim99,
        'SPE_lim': spelim,
        'SPE_lim_99': spelim99,
        'expl_var': mdl.explained_variance_ratio_,
        'sample_names': X.index.tolist(),
        'scaler': scaler,
        'sklearn_model': mdl
    }

    return dict_to_return


"""
Function for transforming new data

Input
--------------------------------
- pcamdl: PCA model trained with pca
- xnew: new data to project in the PCA space (as pandas dataframe)

Output
-----------------------------
- dictionary with: scores (Tnew), SPE
                    
Version
----------------------------
v1 - gb 31/05/2022

"""


def pca_pred(pcamdl, xnew):

    # Extract model
    mdl = pcamdl['sklearn_model']

    # Transform data to numpy array
    x = xnew.to_numpy()

    # Scale new data
    scaler = pcamdl['scaler']
    xs = scaler.transform(x)

    # Transform new data
    tnew = mdl.transform(xs)

    # Calculation of SPE
    xr = mdl.inverse_transform(tnew)
    E = xs - xr
    spe = np.sum(E * E, axis=1)

    # Calculation of T2
    lambdat = np.var(pcamdl['T'], axis=0) ** (-1)
    tsqs = np.sum(np.dot(tnew * tnew, np.diag(lambdat)), axis=1)

    dict_to_return = {
        'Tnew': tnew,
        'T2new': tsqs,
        'SPEnew': spe,
        'sample_names': xnew.index.tolist()
    }

    return dict_to_return


"""
Given the sample ID provides the root sample names and index for groups based on root sample

Inputs
---------------------------
- sampleID: sample names

 Outputs
 --------------------------
 - names: root sample name
 - groups: numeric index dividing samples according to root sample

"""


def groupExtr(sampleID):
    names = list()

    # printing root names of the samples
    for sp in sampleID:
        pos = sp.find('_')
        if pos != -1:
            names.append(sp[0:sp.find('_')])
        else:
            names.append(sp)

    values, groups = unique(names)
    # print(groups)

    return names, groups


"""
Determine unique values and which sample has a specific value

Inputs
---------------------------
- list: list of value to determine unique ones

 Outputs
 --------------------------
 - unique_list: listo of unique values
 - groups: which unique values each sample has
 
"""


def unique(mylist):
    # initialize a null list
    unique_list = []
    element_idx = []

    # traverse for all elements
    for x in mylist:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)

        # find the index of the unique element
        element_idx.append(unique_list.index(x))

    # return the list
    return unique_list, np.array(element_idx, dtype='int32')
