import pandas as pd
from next_flux import next_flux

data_to_predict = pd.read_excel('Example NEXT-FBA Input Process Level Data.xlsx')

main_path = r'C:\Users\rjmor\OneDrive\Documents\PycharmProjects\NEXT-FBA ANN Fully Trained\nn_trained_50'


predicted = next_flux(data_to_predict, main_path, alpha=0.95, exclude_different=False, condition='both',
                      condition_alpha ='95', shsh=False, save_bounds=True)

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


