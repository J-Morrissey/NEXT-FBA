from os.path import join
import pickle
from FluxSet import FluxSet


# ----------------------------------------------------------------------------------------------------------------------
#   INPUT SELECTION
# ----------------------------------------------------------------------------------------------------------------------


########################################################################################################################

# Place here the directory to save NEXT-FBA
main_dir = r"NEXT-FBA"

# Load the data from their path 
# All relevant information will be saved into the FluxSet class
neuralFlux = FluxSet(
    r"Extracellular_uptakes_py.xlsx",
    r"Intracellular_fluxes_py.xlsx",
    r"Extracellular_metadata.xlsx",
    r"Intracellular_metadata.xlsx")

########################################################################################################################


# Extraction of the uptakes list
uptake_list = neuralFlux.extra_metadata_list()

# Selection, extraction and printing of the intracellular reaction to predict
flux_list = neuralFlux.intra_metadata_list()

# List of equilibrium reactions
equilibrium_reactions = neuralFlux.reaction_metadata['Equilibrium'].tolist()

# Cycle over intracellular fluxes to identify the ones to keep and predict
intra_flux = list()
index_to_pop = list()
for i, flx in enumerate(flux_list):
    # Extraction of the data for the selected intracellular flux
    X, Y, sampleID = neuralFlux.dataset_extract(flx)

    # Check if the number of sample is sufficient. Threshold is set to have at least 75% of the sample available
    if len(X) / len(neuralFlux.Xe) >= 0.75:
        intra_flux.append(flx)
    else:
        # Drop equilibrium data for this reaction
        index_to_pop.append(i)

for i in reversed(index_to_pop):
    equilibrium_reactions.pop(i)

# ----------------------------------------------------------------------------------------------------------------------
#   SAVE RESULTS
# ----------------------------------------------------------------------------------------------------------------------

# Save list of inputs
with open(join(main_dir, "uptake_input_list.pkl"), 'wb') as f:
    pickle.dump(uptake_list, f)

# Save list available outputs
with open(join(main_dir, "available_outputs.pkl"), 'wb') as f:
    pickle.dump(intra_flux, f)

# Save list of equilibrium reactions
with open(join(main_dir, "equilibrium_reactions.pkl"), 'wb') as f:
    pickle.dump(equilibrium_reactions, f)

