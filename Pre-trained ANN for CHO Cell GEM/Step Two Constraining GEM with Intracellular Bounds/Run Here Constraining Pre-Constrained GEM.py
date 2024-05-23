import cobra.flux_analysis
import pandas as pd
from next_fba_constrained_model import next_fba_constrained_model

# Input Files:
mapping_table = pd.read_excel('iCHO2441 Mapping.xlsx', index_col=0)
uptake_data = pd.read_excel('example_uptakes_iCHO2441.xlsx', index_col=0)
trained_c13bound_data = pd.read_excel('example_ANN_bounds_iCHO2441.xlsx', index_col=0)
model= cobra.io.read_sbml_model('iCHO2441.xml')
solver = 'cplex'

# Choose experiment
experiment = 'experimentA'

# Setting uptake reaction bounds based on experimental data
for uptake_reaction in list(uptake_data.index):
    uptake_flux_LB = uptake_data.loc[uptake_reaction, experiment + '_LB']
    uptake_flux_UB = uptake_data.loc[uptake_reaction, experiment + '_UB']
    if uptake_reaction in ['biomass_cho', 'biomass_cho_prod', 'ICproduct_Final_demand']:
        model.reactions.get_by_id(uptake_reaction).bounds = (uptake_flux_LB, uptake_flux_UB)
    else:
        model.reactions.get_by_id(uptake_reaction).bounds = (-uptake_flux_UB, -uptake_flux_LB)

constrained_model = next_fba_constrained_model(model,experiment,mapping_table,trained_c13bound_data,solver)


#Example FBA to test constrained_model
constrained_model.objective = 'biomass_cho_prod'
solution = constrained_model.optimize()
optimal_growth = solution.objective_value
print(optimal_growth)