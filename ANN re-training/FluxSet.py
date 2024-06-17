import numpy as np
import pandas as pd
from utils import delete_multiple_element


"""
NEXT-FLUX

Main class containing all C13 data and metadata


Version
-------------------------
v1 - gb 06/06/2022 --> base code

"""


class FluxSet:

    """
    Initialization of the object

    Inputs
    ---------------------------
    - Path with extracellular dataset (rows: samples, columns: uptakes)
    - Path with intracellular dataset (rows: samples, columns: intracellular fluxses)
    - Path with extracellular uptakes metadata
    - Path with intracellular fluxes metadata
    """

    def __init__(self, extra_path, intra_path, e_meta_path, i_meta_path):
        # Initialization of the object
        self.extra_path = extra_path
        self.intra_path = intra_path
        self.e_meta_path = e_meta_path
        self.i_meta_path = i_meta_path

        # actual loading of the data
        self.Xe = pd.read_excel(self.extra_path, header=0, index_col=0)
        self.Xi = pd.read_excel(self.intra_path, header=0, index_col=0)
        self.uptakes_metadata = pd.read_excel(self.e_meta_path, header=0, index_col=0)
        self.reaction_metadata = pd.read_excel(self.i_meta_path, header=0, index_col=0)

        # Extraction of the sample codes and metadata
        self.sampleID = list(self.Xi.index.values)
        self.extra_metadata = list(self.Xe.columns.values)
        self.intra_metadata = list(self.Xi.columns.values)

        # definition of additional attributes
        self.X = []
        self.Y = []

    """
    Preparation of the datasets for further modeling

    Inputs
    ---------------------------
    - Y variable to consider as response 

    Outputs
    --------------------------
    - X
    - Y
    - names of the samples
    """

    def dataset_extract(self, y_var):
        # Method to prepare data for modeling: extraction of x and y
        x = np.array(self.Xe, dtype='float32')
        y = np.array(self.Xi[y_var], dtype='float32')
        names = self.sampleID

        # writing the X and Y in the class
        self.X = x
        self.Y = y

        # identify nan samples
        index = np.where(np.isnan(y))
        x = np.delete(x, index[0], axis=0)
        y = np.delete(y, index[0], axis=0)
        names = delete_multiple_element(names, index[0].tolist())

        return x, y, names

    """
    Extraction of the extracellular data for 

    Inputs
    ---------------------------
    - Y variable to consider as response 

    Outputs
    --------------------------
    - X
    - Y
    - names of the samples
    """

    def prediction_extract(self, y_var):
        # Method to prepare data for modeling: extraction of x and y
        x = np.array(self.Xe, dtype='float32')
        y = np.array(self.Xi[y_var], dtype='float32')
        names = self.sampleID

        # writing the X and Y in the class
        self.X = x
        self.Y = y

        # identify nan samples
        index = np.where(np.isnan(y))
        x = x[index[0]]
        y = y[index[0]]
        names = [nm for i, nm in enumerate(names) if i in index[0]]

        return x, y, names

    """
    Return the intracellular reactions metadata

    Inputs
    --------------------------- 

    Outputs
    --------------------------
    - intracellular reactions metadata
    """

    def intra_metadata_list(self):
        intra_metadata = list(self.Xi.columns.values)

        return intra_metadata

    """
    Return the extracellular uptakes metadata

    Inputs
    --------------------------- 

    Outputs
    --------------------------
    - extracellular uptakes metadata metadata
    """

    def extra_metadata_list(self):
        extra_metadata = list(self.Xe.columns.values)

        return extra_metadata
