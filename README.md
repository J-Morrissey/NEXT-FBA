Neural-net Extracellular Trained Flux Balance Analysis, a hybrid approach to constrain genome-scale models


Summary
NEXT-FBA is a hybrid mechanistic and data-driven model to constrain genome scale models (GEMs).
NEXT-FBA involves training an artificial neural network (ANN) to understand the correlation between exometabolomics and cell metabolism. The ANN predicts intracellular reaction bounds for unseen data and these bounds are used to constrain a GEM. This protocol will walk through two separate procedures. Firstly, how to train the ANN on a new dataset. Secondly, applying a pre-trained ANN to constrain a Chinese hamster ovary (CHO) cell GEM.

For complete details on the use and execution of this protocol, please refer to REF[JM1].

Graphical abstract[JM2]


Before you begin
GEMs represent cellular metabolic reactions to predict and understand metabolic behaviors. Their broad applicability includes cancer biology, metabolic engineering, biopharmaceutical production, and agriculture. Flux balance analysis (FBA) is the predominant method for solving GEMs, predicting reaction fluxes by maximizing or minimizing a metabolic objective. FBA imposes constraints using experimental data like metabolite exchanges to predict intracellular states. Despite advancements, GEMs, particularly for mammalian systems, face challenges due to being underdetermined and having limited constraints, making predictions unreliable.

The integration of exometabolomic data with GEMs can enhance model constraints, yet the complex relationships between external metabolites and internal cell states remain inadequately captured by traditional stoichiometric methods. Hybrid approaches combining mechanistic and data-driven models address these limitations by leveraging machine learning's predictive power alongside traditional models' biological insights. NEXT-FBA (Neural-net EXtracellular Trained Flux Balance Analysis), uses an ANN trained on CHO cell data to predict bounds for intracellular reactions. These bounds constrain a CHO cell GEM, accurately predicting metabolic states with minimal data. NEXT-FBA demonstrates superior performance in predicting intracellular flux distributions over other traditional constraining methods [ref[JM3]].

This protocol outlines two steps in utilizing NEXT-FBA. The first step is if the user wishes to re-train the ANN on new datasets, either a completely new system (e.g. yeast, E. Coli. HEK-293) or adding data to the existing CHO cell database. This step is optional if the user doesn’t wish to retrain the ANN. The second step deploys a pre-trained ANN to predict intracellular bounds and constrain a CHO cell GEM. 

Download NEXT-FBA Toolkit

Timing: 5 minutes

1. Download NEXT-FBA files from https://github.com/J-Morrissey/NEXT-FBA (FIGURE). Click the green ‘Code’ button at the upper right corner and download the toolkit and the example dataset as a zip file by clicking ‘Download Zip’.

Figure 1: Screenshot of the NEXT-FBA GitHub page from which the files can be downloaded
[JM4]


Set up python working environment

Timing: 5 minutes

2. Open Python.
Users must download and install Python beforehand. A Python integrated development environment (IDE), such as Pycharm, is recommended. 
3. Unzip NEXT-FBA files into the working directory (Pycharm Project).


Formatting of user generated data

Timing: Variable (5 minutes if just using pre-trained ANN, 30 minutes for formatting data to train ANN with new data).

4. Formatting new dataset to train the ANN. Required training data are organized in four Excel files. 
a. Process-level data (exometabolomics): This file should include measurements of metabolite uptakes from the extracellular environment, growth rate and productivity. Organize the spreadsheet with observations labels along the first column and metabolite exchanges labels along the first row (example in Figure 1). The units for exchanges and productivity are mmol gDCW-1hr-1 and growth rate in hr-1, where positive exchange means metabolite uptake and negative is secretion. This dataset can contain missing values as ‘nan’. 
Figure 1. Demonstration of process-level data to input for the training of ANN.
b. Intracellular flux data: This file should contain measurements of flux of intracellular reactions, recorded for each observation. Organize the spreadsheet with observation labels along the first column and intracellular reaction labels along the first row (example in Figure 2). The units of measurements for intracellular fluxes are ###, where the positive values indicate flux directed according to reaction expression and negative opposite to reaction expression. This dataset can contain missing values as ‘nan’.
Figure 2.  Demonstration of intracellular flux data to input for the training of ANN.
Note: Both exometabolomics and intracellular fluxes datasets should contain three data points for each experimental run, corresponding to the lower bound (LB), median (MEDIAN) and upper bound (UB) of the measured values.
c. Exometabolomics metadata: This file contains additional information related to process-level data, such as reaction equations, metabolite reference codes, etc. Organize the spreadsheet with labels of metadata information along the first column and extracellular metabolite exchange labels along the first row (example in Figure 3). Any supplementary information regarding the extracellular exchanges can be included in this data file.
[JM5]Figure 3. Demonstration of process-level metadata
d. Intracellular flux metadata: This file contains additional information related to intracellular reactions, such as reaction expression and reversibility. Organize the spreadsheet with intracellular reaction labels along the first column, and the labels of metadata information along the first row (example in Figure 4). Ensure the presence of a column named ‘Equilibrium’, flagging reversible reactions (assign 1 for reversible reactions and 0 for irreversible ones). This file can include any supplementary information regarding intracellular reactions.

Figure 4. Demonstration of intracellular reaction metadata.
5. Formatting user generated process datasets for pre-trained ANN.
a. Input dataset into pre-trained ANN (data_to_predict): This dataset is used as an input into the ANN to make intracellular reaction bound predictions. A demonstration dataset 'Example NEXT-FBA Input Process Level Data.xlsx' is provided and should be replaced with a user dataset. Metabolite exchanges plus growth (biomass) and productivity (Antibody)  hould be column labels with experiments in the rows, as shown in FIGURE. The units for exchanges and productivity are mmol gDCW-1hr-1 and growth rate in hr-1, where positive exchange means metabolite uptake and negative is secretion. The file can contain missing data and this will be automatically inputted into the ANN, see REF for more details. 

b. Input dataset to constrain [JM6]GEM (uptake_data): This dataset contains the metabolite exchange data (uptake and secretions) that are used to constrain exchange reactions in the model. The dataset contains the same information as data_to_predict, it is just formatted to suit the GEM of choice. It is likely that the user has already formatted this dataset to fit their own constraining code, but a demonstration dataset for the iCHO2441 GEM, ‘example_uptakes_iCHO2441.xlsx’, is also provided. Row indexes are the iCHO2441 GEM name for the exchange reaction, and a column for the exchange lower bound, upper bound and median, as shown in FIGURE. The units for exchanges and productivity are mmol gDCW-1hr-1 and growth rate in hr-1, where positive exchange means metabolite uptake and negative is secretion. 


Figure 3: Demonstration process level dataset to constrain your GEM. This contains exactly the same information as FIGURE, but allows the user to input lower and upper bounds, as well as bounds for additional metabolites not covered in FIGURE.
Key resources table

Key resources table
REAGENT or RESOURCESOURCEIDENTIFIERDeposited dataNEXT-FBA Master FolderThis articlehttps://github.com/J-Morrissey/NEXT-FBAANN training data (optional). If the user wishes to re-train the ANN with new data, they must provide the extracellular exchange and the corresponding intracellular fluxes (from 13C fluxomics or elsewhere)User providedN/AUser provided process data to use in pre-trained ANN. This includes exchange data for metabolites (20 amino acids, lactate glucose), as well as growth rate and recombinant protein productivity, if appropriate.User provided. Demonstration file available in GitHub.https://github.com/J-Morrissey/NEXT-FBASoftware and algorithmsPython 3.0Python Software Foundationhttps://www.python.org/downloads/COBRApyPython PackageREF[JM7]PandasPython Packagehttps://pandas.pydata.org/NumpyPython Packagehttps://numpy.org/TensorflowPython Packagehttps://www.tensorflow.orgScikit-learnPython Packagehttps://scikit-learn.org/stable/ScipyPython Packagehttps://scipy.org/MatplotlibPython Packagehttps://matplotlib.org/CPLEXIBMhttps://www.ibm.com/products/ilog-cplex-optimization-studio

Step-by-step method details

Here, the described step-by-step methods for re-training the NEXT-FBA ANN with user-provided datasets (Steps 1-3) and applying a trained ANN to a CHO cell GEM (Steps 4-5) are covered. Steps 1-3 are optional if the user doesn’t wish to re-train the ANN and can immediately deploy the pre-trained ANN. Steps 4-5 can use either the pre-trained ANN from the master folder, or if the user is working with CHO cell systems, it can use the newly-trained ANN from Steps 1-3. 

Steps 4 and 5 are separated as it allows the user to take the intracellular bound predictions from Step 4 and apply them to another stoichiometric model other than iCHO2441. However this would require a new back mapping file (similar to ‘iCHO2441 Mapping.xlsx’) and additional modifications not covered in this protocol. Likewise, if the user has trained the ANN with data from another cell system (e.g. E. Coli, HEK-293, yeast), this would require modifications to Steps 4-5 not covered in this protocol.

Training ANN on new data

Timing[JM8]: variable [s, min, h, days, or weeks]

This step trains the ANN on new user-provided datasets and sets up all requirements to predict intracellular flux bounds used to constrain GEMs (Step Two).

1. Use the Python script “train_deploy_network_singlevariable.py” to train the ANN on user-provided datasets. This step should be repeated for each intracellular reaction available in the study.
a.  Specify the name of the main directory where the trained ANN will be stored. This directory will contain the ANN models trained to predict the flux of each intracellular reaction.  
          >main_dir = r”Next-FBA”
b. Choose the intracellular flux to be utilized for ANN training by setting the index of the desired intracellular reaction. 
Note: In Python, indices start from 0 and range up to the total number of available intracellular reactions minus 1.
          >variable_to_predict = 1
c. Configure the inputs for data augmentation.
i. Enable data augmentation by setting “do_smote” to “True”; set to “False” otherwise. 
ii. Specify the number of new observations to generate with SMOTE for each original observation using “sample_to_augment”.
iii. Specify the number of nearest neighbour observations to consider for SMOTE data augmentation with “neighbors_aug”.
iv. Specify the number of observations to generate by adding white noise to each input observation using “noiseDAsamples”. 
v. Set the amount of noise to add to the exometabolomics matrix with “x_noise” (specified as a percentage normalized to 1). 
vi. Set the amount of noise to add to the intracellular flux matrix with “y_noise” (specified as a percentage normalized to 1).

d. Specify the number of different ANN initializations to evaluate. The ANN exhibiting the lowest final loss value among these iterations will be chosen.
          >n_models = 25
e. Specify the file paths for input data. The paths of the four Excel files are provided as inputs to the “FluxSet” class, which manages the data. 
Note: Please provide the absolute path for the data files.

f. Specify for each intracellular reaction the activation function that will be used in the ANN layers. Populate the “activation_list” list with the chosen activation function (either “relu” or “tanh”) for each intracellular reactions under analysis. 
Note: Optimal activation function should be selected based on model performance in cross-validation.
g. Establish the search limits for optimizing the ANN hyperparameters. The ”nn1” array specifies the number to explore for neurons of the first hidden layer. The “nn2” array specifies the number of neurons to explore for the second hidden layer. The “lr” array specifies the different learning rates to explore.

h. Configure parameters for ANN cross-validation. “no_kfold_split” specifies the number of splits for k-fold cross-validation used to identify the optimal hyperparameters. “max_epochs” specifies the maximum number of training epochs during cross-validation. “cv_iterations” specifies the number of iterations for the Monte Carlo cross-validation used to identify the optimal number of training epochs. “mc_test_fraction” specifies the fraction of observation allocated to the validation set within the Monte Carlo cross-validation process.

i. Execute the Python script to initiate ANN training.
j. The trained ANN for the chosen intracellular reaction is stored in a folder named “deployed_nn_#yourreactionname#” within the main NEXT-FBA directory (specified at step a). All additional necessary data is saved in the “utils” folder. 
Note: Training time for the ANN varies depending on the hardware used. High-performance PCs may be required for faster training.
2.  Use the Python script “train_pca.py” to generate the PCA model necessary for assessing the similarity of new observations to the training dataset. 
a. Specify the name of the main NEXT-FBA directory where the trained ANN is stored. The trained PCA model will be saved in this directory.
          >main_dir = r”Next-FBA”
b. Specify the file path for input data. The paths of the four Excel files are provided as inputs to the “FluxSet” class, which manages the data. 
Note: Please provide the absolute path for the data files.

c. Configure parameters for PCA cross-validation, which are used to determine the optimal number of principal components (PCs). “max_pcs” defines the maximum number of PCs to test during cross-validation, while “cross_val_split” sets the number of k-fold data splits for cross-validation.

d. Execute the Python script.
e. Review the results of the PCA cross-validation displayed in your Python terminal. Determine the optimal number of PCs either by selecting the configuration that minimizes the Root Mean Squared Error of Cross-Validation (RMSECV) or by employing the eigenvalue greater than one rule.

Figure ##. Example of the PCA cross-validation results displayed in your Python terminal. In this example, 5 PCs have been selected according to the eigenvalue greater than one rule.
f. Set the number of PCs determined in step 5 as “ncomp” in the script.
          >ncomp = 5
g. Execute the Python script. Note: With high number of observations or extensive process-level data, computational time may increase.
h. The trained PCA model will be stored in the main NEXT-FBA directory. Additionally, plots depicting PCA scores (for the first and second PCs), an outlier map, and PCA loadings are generated. These images are not automatically saved, but users can do if desired. 

(a)							(b)

(c)
Figure ##. Example of plots depicting the PCA scores (a), the PCA outlier map (b), and the PCA loadings (c).

3. Use the Python script “create_flux_list.py” to generate the metadata which are essential for the ANN to predict reaction bounds.
a. Specify the name of the main NEXT-FBA directory where the trained ANN is stored. All generated metadata will be saved in this directory.
          >main_dir = r”Next-FBA”
b. Specify the file path for input data. The paths of the four Excel files are provided as inputs to the “FluxSet” class, which manages the data. 
Note: Please provide the absolute path for the data files.

c. Execute the Python script.
d. Files containing exometabolomic input, available intracellular reactions, and reversible intracellular reactions are stored into the main NEXT-FBA directory.

Constraining CHO Cell GEM using pre-trained ANN

Timing: 20 minutes

This step uses a trained ANN (either from previous step or from provided pre-trained ANN), to predict intracellular flux bounds using user-provided process data (Step 4). These intracellular bounds are then used to constrain the iCHO2441 GEM [ref] (Step 5).

4. Use the python script ‘Run Here ANN Intracellular Predictions.py’ , from the folder ‘Step One Predicting Intracellular Bounds with Pre-trained ANN’ to predict intracellular reaction bounds from the pre-trained ANN.
a.  Modify the inputs to the ‘next_flux’ function. TABLE outlines each input.




Table 1: Inputs and output for pre-trained ANN
InputDescriptiondata_to_predictUser provided process data, formatted as outlined in SECTION. This is a pandas.DataFrame with row and column index provided.main_path
Path of the directory containing NEXT-FLUX files.alphaConfidence level of NN predicted intracellular fluxes. It is used to determine LB and UB.conditionRationale to define an experiment different from the training one. It is based on PCA projection.
* 'T2': experiments outside Hotelling's T2 limit are excluded
* 'SPE': experiments outside SPE limit are excluded
* 'all': only experiments inside both limits are included
* 'both' experiments outside both limits are excludedcondition_alphaSelect the confidence level of PCA diagnostics ('95'/'99')shshStop printing of info (True/False)save_boundsFlag for saving the predicted bounds as Excel file in the current directory (True/False)save_similarityFlag for saving the similarity scores as Excel file in the current directory (True/False)Output Descriptionpredicted_boundsIntracellular bounds predictions for 43 reactions in central carbon metabolism. This is used in the following steps to constrain a CHO cell GEM.similaritySimilarity scores for new experiments
b. Run the script. This will generate two files.
i. A ‘predicted_bounds’ excel file with the date and time. This file contains the predicted intracellular flux bounds used in Step 5.
ii. A ‘similarity’ excel file with the date and time. This file will provide the similarity scores between the process data from the new experiment and the experiments used to train the ANN. It gives the distance from average and the difference in correlation. If these values are above 1 for any experiment, it differs significantly from the training dataset, and hence intracellular flux predictions may be unreliable.
5. Use[JM9] the python script ‘Run Here Constraining GEM.py’ to constrain the CHO cell GEM with predicted intracellular bounds. 
a. Modify the inputs to the ’next_fba_constrained_model’ function. Table outlines each input.



Table 2: Inputs and outputs to next_fba_constrained_model function.
InputDescriptionmodelCOBRApy metabolic model to constrainexperimentExperiment from which to constrain model. This will select the correct bounds for exchange data and predicted bounds from ANN in Part One. mapping_tableBack mapping table from NEXT-FBA reactions to iCHO2441 reactions. Provided in files.trained_c13bound_dataPredicted bounds from ANN in Part OneOutputDescriptionconstrained_modelNEXT-FBA constrained COBRApy metabolic model.
b. Run the script.
i. This constrains ‘model’ to become ‘constrained_model’, a COBRApy model constrained with NEXT-FBA intracellular bounds.
ii. This model is ready to be used for flux analysis, e.g. FBA or flux sampling. An example FBA growth maximization problem is given at the end of the script.

[JM1]NEXT-FBA paper
[JM2]Do this
[JM3]Our paper
[JM4]Update this
[JM5]Does this file need rows 3 and 4? This is specific to CHOmpact model so it would be best if we could keep it general
[JM6]Check the wording here to see if this is necessary
[JM7]COBRApy: COnstraints-Based Reconstruction and Analysis for Python
[JM8]Provide an estimate here please
[JM9]Check numbering at end
