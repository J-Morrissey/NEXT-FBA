import cobra.flux_analysis
from cobra.sampling import sample
import pandas as pd
from next_fba_constrained_model import next_fba_constrained_model

# Input Files:
mapping_table = pd.read_excel('iCHO2441 Mapping.xlsx', index_col=0)
uptake_data = pd.read_excel('test.xlsx', index_col=0)
trained_c13bound_data = pd.read_excel('example_ANN_bounds_iCHO2441.xlsx', index_col=0)
model= cobra.io.read_sbml_model('iCHO2441.xml')
solver = 'cplex'

# Choose experiment
experiment = 'experimentA'

constrained_model = next_fba_constrained_model(model,experiment,mapping_table,
                                               uptake_data,trained_c13bound_data,solver)

constrained_model.objective = 'biomass_cho_prod'
solution = constrained_model.optimize()
optimal_growth = solution.objective_value
print(optimal_growth)

# Sampling
flux_samples = sample(constrained_model, 1, processes=1, thinning=1)
flux_samples = flux_samples.transpose()
flux_samples.to_excel('NEXT-FBA_samples_' + experiment + '.xlsx')



