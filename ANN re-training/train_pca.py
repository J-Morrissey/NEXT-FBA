from os.path import join
import pickle
import numpy as np
from next_pca import pca
from FluxSet import FluxSet
from utils import groupExtr
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------
#   INPUT SELECTION
# ----------------------------------------------------------------------------------------------------------------------


########################################################################################################################

# Place here the main directory to save NEXT-FBA
main_dir = r"NEXT-FBA"

# Load the data from their path.
# All relevant information will be saved into the FluxSet class
neuralFlux = FluxSet(
    r"Extracellular_uptakes_py.xlsx",
    r"Intracellular_fluxes_py.xlsx",
    r"Extracellular_metadata.xlsx",
    r"Intracellular_metadata.xlsx")
    
# Set input for PCA cross-validation
# This is required to set the best number of PCs
max_pcs = 10
cross_val_split = 8


########################################################################################################################


# Selection, extraction and printing of the intracellular reaction to predict
flux_list = neuralFlux.intra_metadata_list()
print('The selected intracellular flux is: {}'.format(flux_list[0]))

# Extraction of the data for the selected intracellular flux
X, Y, sampleID = neuralFlux.dataset_extract(flux_list[0])

# Extraction of the root names and group indices for all the available samples
sample_names, group_idx = groupExtr(sampleID)


# ----------------------------------------------------------------------------------------------------------------------
#   BUILD PCA
# ----------------------------------------------------------------------------------------------------------------------

# Cross-validate PCA on exometabolomics
cv_pca = pca(X, max_pcs, cross_val=True, cross_val_split=cross_val_split, 
            sample_groups=group_idx, shsh=False, alpha=0.95)


########################################################################################################################
# SELECT THE NUMBER OF PRINCIPAL COMPONENTS - Minimum RMSECV

# Number of Principal Components 
# Should be found as the one minimizing
# RMSECV in cross-validation
ncomp = 5

########################################################################################################################


# Build pca model
pca_model = pca(X, ncomp, cross_val=False)


# ----------------------------------------------------------------------------------------------------------------------
#   SAVE RESULTS
# ----------------------------------------------------------------------------------------------------------------------

# Save pca model
with open(join(main_dir, "C13_pca_trained_model.pkl"), 'wb') as f:
    pickle.dump(pca_model, f)
    
   

# ----------------------------------------------------------------------------------------------------------------------
#   VISUALIZATION OF PCA MODEL
# ----------------------------------------------------------------------------------------------------------------------

# SET VALUES TO VISUALIZE
plot_scores = [1, 2]
plot_weights = 1
    
    
if ncomp >= 2:
    # Score plot
    plt.figure()
    plt.scatter(pca_model['T'][:, plot_scores[0]-1], pca_model['T'][:, plot_scores[1]-1], s=50,
                marker="s")  # C13 data
    # plt.scatter(predmdl['Tnew'][:, plot_scores[0]-1], predmdl['Tnew'][:, plot_scores[1]-1], s=40)  # new data

    # Plot origin lines
    plt.plot([plt.xlim()[0], plt.xlim()[1]], [0, 0], color='black', linewidth=0.9)
    plt.plot([0, 0], [plt.ylim()[0], plt.ylim()[1]], color='black', linewidth=0.9)

    plt.xlabel(f"scores component {plot_scores[0]} (R2x={np.round(100*pca_model['expl_var'][plot_scores[0]-1], 1)} %)")
    plt.ylabel(f"scores component {plot_scores[1]} (R2x={np.round(100*pca_model['expl_var'][plot_scores[1]-1], 1)} %)")

    plt.show()
    

# Outlier map
plt.figure()
plt.scatter(pca_model['T2'], pca_model['SPE'], s=50, marker="s")  # C13 data

# Plot origin lines
plt.plot([pca_model['T2_lim'], pca_model['T2_lim']], [0, plt.ylim()[1]], color='black', linewidth=0.9)
plt.plot([0, plt.xlim()[1]], [pca_model['SPE_lim'], pca_model['SPE_lim']], color='black', linewidth=0.9)

plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])

plt.xlabel(f"Hotelling's T2 (R2x={np.round(100*np.sum(pca_model['expl_var']), 1)} %)")
plt.ylabel("SPE")

plt.show()



# PCA Loadings
plt.figure(figsize=(10, 17))
plt.bar(np.arange(len(pca_model['P'])), pca_model['P'][:, plot_weights-1])

plt.xlabel('uptakes')
plt.ylabel(f"weights of component {plot_weights}")
plt.xticks(np.arange(len(pca_model['P'])), labels=neuralFlux.extra_metadata, rotation=90)

plt.show()
