import cobra.flux_analysis
from cobra.sampling import sample
import pandas as pd

# Created by James Morrissey on 10/04/24.
# This script performs NEXT-FBA for constraining iCHO2441
# Reaction bounds from the ANN are mapped to a GEM, then the number of feasible bounds are maximised using a MILP problem

### INPUTS ###
# 1. model - iCHO2441 GEM
# 2. experiment - Experiment for which to constrain
# 3. mapping table - Mapping table to map NEXT-FBA reactions to iCHO2441
# 4. uptake_data - Metabolite exchange data to constrain iCHO2441
# 5. trained_c13bound_data - Upper and lower bounds from ANN

### OUTPUT ###
# 1. Constrained GEM

### OTHER REQUIREMENTS ###
#CPLEX solver for MILP

#NEXT-FBA Function
def next_fba_constrained_model(model,experiment,mapping_table,trained_c13bound_data,solver):
    # Extracting NEXT-FBA constrained reaction names
    next_flux_constrained_reactions = [x for x in list(mapping_table['iCHO2441 Name']) if x != 'none']

    # Finding and renaming duplicate reactions
    dupes = [x for x in next_flux_constrained_reactions if next_flux_constrained_reactions.count(x) > 1]
    for reaction in dupes:
        count = next_flux_constrained_reactions.count(reaction)
        if count > 1:
            for i in range(count - 1):
                index = [i for i, n in enumerate(next_flux_constrained_reactions) if n == reaction][i]
                next_flux_constrained_reactions[index] = reaction + '_duplicate' + str(i + 1)

    # Any unwanted C13 reactions can be removed here
    unwanted_reactions = []

    bounded_c13reactions = list(trained_c13bound_data.columns)
    bounded_c13reactions = [i.replace('LB_', '') for i in bounded_c13reactions]
    bounded_c13reactions = [i.replace('MEDIAN_', '') for i in bounded_c13reactions]
    bounded_c13reactions = [i.replace('UB_', '') for i in bounded_c13reactions]
    bounded_c13reactions = list(dict.fromkeys(bounded_c13reactions))
    bounded_c13reactions = [reaction for reaction in bounded_c13reactions if reaction not in unwanted_reactions]

    # Finding index of C13 reactions to split mapping table later on
    all_reaction_names = [x for x in list(mapping_table['Enzyme']) if str(x) != 'nan']
    reaction_indexes = []
    for name in all_reaction_names:
        reaction_indexes.append(list(mapping_table['Enzyme']).index(name))
    reaction_indexes.append(len(list(mapping_table['Enzyme'])))

    ### MODEL BUILDING ###
    ### BACKMAPPING REACTIONS TO FIND INTRACELLULAR BOUNDS ###
    # Initializing lists to find the lower bounds (LB) and upper bounds (UB) for each reaction in the GEM
    lowerbounds = []
    upperbounds = []
    # Initializing a list to collate information on parallel reactions in a dictionary to create summed constraints later on
    parallel_reactions = []
    milp_objective_coefficients = []
    # Looping through all C13 reactions to find bounds for all intracellular reactions
    for i, name in enumerate(all_reaction_names):
        # Splitting the mapping table into segments for each C13 reaction
        reaction_segment = mapping_table.iloc[reaction_indexes[i]:reaction_indexes[i + 1]][:]
        # Lists for parallel reaction dictionary
        parallel_names = []
        parallel_directionality = []
        parallel_indexes = [x for x in range(len(list(reaction_segment.iloc[:]['Pathway']))) if
                            list(reaction_segment.iloc[:]['Pathway'])[x] != list(
                                reaction_segment.iloc[:]['Pathway'])[x - 1]]
        parallel_indexes.insert(0, 'placeholder')
        number_series = list(reaction_segment.iloc[:]['Pathway']).count('s')
        without_s = [x for x in list(reaction_segment.iloc[:]['Pathway']) if x != 's']

        if number_series == len(list(reaction_segment.iloc[:]['Pathway'])):
            number_independent_parallel = 0
        else:
            number_independent_parallel = max(1, len(
                [x for x in range(len(without_s)) if without_s[x] != without_s[x - 1]]))

        milp_objective_coefficient = 1 / (number_series + number_independent_parallel)

        # Looping through all the GEM reactions for a given C13 reaction
        for j in range(len(reaction_segment['iCHO2441 Name'])):
            parallel = False
            parallel_reaction = {}
            if name not in bounded_c13reactions or str(reaction_segment.iloc[j]['Pathway']) == 'nan' or str(
                    trained_c13bound_data['LB_' + name][experiment]) == 'nan':
                lowerbounds.append('nan')
                upperbounds.append('nan')
                milp_objective_coefficients.append('nan')
            else:
                c13lowerbound = trained_c13bound_data['LB_' + name][experiment] * 1 / float(
                    reaction_segment.iloc[j]['Direction'])
                c13upperbound = trained_c13bound_data['UB_' + name][experiment] * 1 / float(
                    reaction_segment.iloc[j]['Direction'])
                if reaction_segment.iloc[j]['Pathway'] == 's':
                    lowerbounds.append(min(c13lowerbound, c13upperbound))
                    upperbounds.append(max(c13lowerbound, c13upperbound))
                    milp_objective_coefficients.append(milp_objective_coefficient)
                if 'p' in reaction_segment.iloc[j]['Pathway']:
                    if j in parallel_indexes:
                        parallel_reactions.append(parallel_reaction)
                        parallel_names = []
                        parallel_directionality = []
                    parallel = True
                    parallel_names.append(reaction_segment.iloc[j]['iCHO2441 Name'])
                    parallel_directionality.append(reaction_segment.iloc[j]['Direction'])
                    parallel_reaction['Parallel Order'] = reaction_segment.iloc[j]['Pathway']
                    parallel_reaction['C13 Name'] = name
                    parallel_reaction['Model Names'] = parallel_names
                    parallel_reaction['Model Directionality'] = parallel_directionality
                    parallel_reaction['LB'] = trained_c13bound_data['LB_' + name][experiment]
                    parallel_reaction['UB'] = trained_c13bound_data['UB_' + name][experiment]
                    parallel_reaction['MILP Coefficent'] = milp_objective_coefficient

                    lowerbounds.append(min(c13lowerbound, c13upperbound, -c13lowerbound, -c13upperbound))
                    upperbounds.append(max(c13lowerbound, c13upperbound, -c13lowerbound, -c13upperbound))
                    milp_objective_coefficients.append(0)

                if parallel == True and parallel_indexes[-1] == 'placeholder' or parallel_indexes[-1] == j:
                    if parallel_reaction in parallel_reactions or parallel_reaction == {}:
                        pass
                    else:
                        parallel_reactions.append(parallel_reaction)
        i += 1

    mapping_table['LB'] = lowerbounds
    mapping_table['UB'] = upperbounds
    mapping_table['Objective Coefficient'] = milp_objective_coefficients

    ### MODEL BUILDING ###
    model.solver = solver

    ### STARTING MILP TO FIND MAX FEASIBLE NEXT-FLUX CONSTRAINTS ###
    max_cns_model = model.copy()

    # Create dictionaries for NEXT-FLUX bounds and original bounds
    lowerbounds_dict = dict(zip(next_flux_constrained_reactions, lowerbounds))
    upperbounds_dict = dict(zip(next_flux_constrained_reactions, upperbounds))
    og_lowerbounds = []
    og_upperbounds = []
    reaction_ids = []

    for reaction in max_cns_model.reactions:
        reaction = str(reaction)
        colon_loc = reaction.find(':')
        id = reaction[:colon_loc]
        reaction_ids.append(id)
        og_lowerbounds.append(max_cns_model.reactions.get_by_id(id).lower_bound)
        og_upperbounds.append(max_cns_model.reactions.get_by_id(id).upper_bound)
    og_lowerbounds_dict = dict(zip(reaction_ids, og_lowerbounds))
    og_upperbounds_dict = dict(zip(reaction_ids, og_upperbounds))

    # Create dict of MILP objective coefficients
    milp_objective_coefficients = dict(zip(next_flux_constrained_reactions, milp_objective_coefficients))

    # Create dictionary for binary variables and constraints
    binary_variables = {}
    binary_constraints_LB = {}
    binary_constraints_UB = {}
    forced_binary = {}
    attemped_bounds = []

    # Setting up binary variables and constraints for NEXT-FLUX
    for reaction in next_flux_constrained_reactions:
        if lowerbounds_dict[reaction] == 'nan':
            continue
        elif '_duplicate' in reaction:
            index = reaction.index('_duplicate')
            reaction_real_name = reaction[:index]
            attemped_bounds.append(reaction)
            binary_variables['x_' + reaction] = max_cns_model.problem.Variable('x_' + reaction, type="binary")
            binary_constraints_UB['binary_UB_' + reaction] = max_cns_model.problem.Constraint(
                max_cns_model.reactions.get_by_id(reaction_real_name).flux_expression -
                upperbounds_dict[reaction] * binary_variables['x_' + reaction] -
                og_upperbounds_dict[reaction_real_name] * (1 - binary_variables['x_' + reaction]), lb=-2000, ub=0)
            binary_constraints_LB['binary_LB_' + reaction] = max_cns_model.problem.Constraint(
                max_cns_model.reactions.get_by_id(reaction_real_name).flux_expression -
                lowerbounds_dict[reaction] * binary_variables['x_' + reaction] -
                og_lowerbounds_dict[reaction_real_name] * (1 - binary_variables['x_' + reaction]), lb=0, ub=2000)
            max_cns_model.add_cons_vars(
                [binary_variables['x_' + reaction], binary_constraints_LB['binary_LB_' + reaction],
                 binary_constraints_UB['binary_UB_' + reaction]])
        else:
            attemped_bounds.append(reaction)
            binary_variables['x_' + reaction] = max_cns_model.problem.Variable('x_' + reaction, type="binary")
            binary_constraints_UB['binary_UB_' + reaction] = max_cns_model.problem.Constraint(
                max_cns_model.reactions.get_by_id(reaction).flux_expression -
                upperbounds_dict[reaction] * binary_variables['x_' + reaction] -
                og_upperbounds_dict[reaction] * (1 - binary_variables['x_' + reaction]), lb=-2000, ub=0)
            binary_constraints_LB['binary_LB_' + reaction] = max_cns_model.problem.Constraint(
                max_cns_model.reactions.get_by_id(reaction).flux_expression -
                lowerbounds_dict[reaction] * binary_variables['x_' + reaction] -
                og_lowerbounds_dict[reaction] * (1 - binary_variables['x_' + reaction]), lb=0, ub=2000)
            max_cns_model.add_cons_vars(
                [binary_variables['x_' + reaction], binary_constraints_LB['binary_LB_' + reaction],
                 binary_constraints_UB['binary_UB_' + reaction]])

    # Binary variables and constraints for parallel reactions
    summed_reactions = {}
    attemped_parallel_bounds = []
    for i in range(len(parallel_reactions)):
        parallel_reaction = parallel_reactions[i]['C13 Name']
        parallel_order = parallel_reactions[i]['Parallel Order']
        attemped_parallel_bounds.append(parallel_reaction + parallel_order)
        binary_variables['x_parallel_' + parallel_reaction + parallel_order] = max_cns_model.problem.Variable(
            'x_parallel_' + parallel_reaction + parallel_order, type="binary")
        binary_constraints_UB[
            'binary_UB_parallel_' + parallel_reaction + parallel_order] = max_cns_model.problem.Constraint(
            sum(
                max_cns_model.reactions.get_by_id(parallel_reactions[i]['Model Names'][x]).flux_expression /
                parallel_reactions[i]['Model Directionality'][x] for x in
                range(len(parallel_reactions[i]['Model Names']))) -
            parallel_reactions[i]['UB'] * binary_variables['x_parallel_' + parallel_reaction + parallel_order] -
            1000 * (1 - binary_variables['x_parallel_' + parallel_reaction + parallel_order]), lb=-2000, ub=0)
        binary_constraints_LB[
            'binary_LB_parallel_' + parallel_reaction + parallel_order] = max_cns_model.problem.Constraint(
            sum(
                max_cns_model.reactions.get_by_id(parallel_reactions[i]['Model Names'][x]).flux_expression /
                parallel_reactions[i]['Model Directionality'][x] for x in
                range(len(parallel_reactions[i]['Model Names']))) -
            parallel_reactions[i]['LB'] * binary_variables['x_parallel_' + parallel_reaction + parallel_order] +
            1000 * (1 - binary_variables['x_parallel_' + parallel_reaction + parallel_order]), lb=0, ub=2000)
        max_cns_model.add_cons_vars(
            [binary_variables['x_parallel_' + parallel_reaction + parallel_order],
             binary_constraints_LB['binary_LB_parallel_' + parallel_reaction + parallel_order],
             binary_constraints_UB['binary_UB_parallel_' + parallel_reaction + parallel_order]])

    # Create objective function
    max_cns_model.objective = max_cns_model.problem.Objective(
        sum([binary_variables['x_' + reaction] * milp_objective_coefficients[reaction] for reaction in
             attemped_bounds]) +
        sum([binary_variables['x_parallel_' + attemped_parallel_bounds[i]] * parallel_reactions[i]['MILP Coefficent']
             for i in range(len(attemped_parallel_bounds))]), direction='max')

    # Optimise the model
    solution = max_cns_model.optimize()
    objective_value = solution.objective_value
    #print('COBRApy Objective Value:' + str(objective_value))

    optimal_binary_variables = {}
    # Assigning optimal binary variables
    for reaction in attemped_bounds:
        if '_duplicate' in reaction:
            index = reaction.index('_duplicate')
            reaction_real_name = reaction[:index]
            value = max_cns_model.solver.variables['x_' + reaction].primal
            optimal_binary_variables[reaction] = value
        else:
            value = max_cns_model.solver.variables['x_' + reaction].primal
            optimal_binary_variables[reaction] = value

    for reaction in attemped_parallel_bounds:
        test = 'x_parallel_' + reaction
        first = getattr(max_cns_model.solver.variables, test)
        second = getattr(first, 'primal')
        optimal_binary_variables[reaction] = second

    ### Constraining the original model ###
    # Constrain intracellular reactions, ignoring duplicates
    for reaction in next_flux_constrained_reactions:
        if lowerbounds_dict[reaction] != 'nan' and '_duplicate' in reaction and optimal_binary_variables[reaction] == 1:
            pass
        elif lowerbounds_dict[reaction] != 'nan' and optimal_binary_variables[reaction] == 1:
            model.reactions.get_by_id(reaction).lower_bound = lowerbounds_dict[reaction]
            model.reactions.get_by_id(reaction).upper_bound = upperbounds_dict[reaction]

    # Considering duplicates by choosing the most restrictive bounds
    duplicate_constraints_LB = {}
    duplicate_constraints_UB = {}
    for reaction in next_flux_constrained_reactions:
        if lowerbounds_dict[reaction] != 'nan' and '_duplicate' in reaction and optimal_binary_variables[reaction] == 1:
            index = reaction.index('_duplicate')
            reaction_real_name = reaction[:index]
            if reaction_real_name not in duplicate_constraints_LB:
                duplicate_constraints_LB[reaction_real_name] = []
                duplicate_constraints_LB[reaction_real_name].append(lowerbounds_dict[reaction_real_name])
                duplicate_constraints_UB[reaction_real_name] = []
                duplicate_constraints_UB[reaction_real_name].append(upperbounds_dict[reaction_real_name])
            duplicate_constraints_LB[reaction_real_name].append(lowerbounds_dict[reaction])
            duplicate_constraints_UB[reaction_real_name].append(upperbounds_dict[reaction])
            model.reactions.get_by_id(reaction_real_name).lower_bound = max(
                [x for x in list(duplicate_constraints_LB[reaction_real_name]) if x != 'nan'])
            model.reactions.get_by_id(reaction_real_name).upper_bound = min(
                [x for x in list(duplicate_constraints_UB[reaction_real_name]) if x != 'nan'])

    # Bounding parallel reactions
    summed_reactions = {}
    for i in range(len(parallel_reactions)):
        parallel_reaction = parallel_reactions[i]['C13 Name']
        parallel_order = parallel_reactions[i]['Parallel Order']
        if optimal_binary_variables[parallel_reaction + parallel_order] == 1:
            summed_reactions[parallel_reactions[i]['C13 Name']] = model.problem.Constraint(
                sum(model.reactions.get_by_id(parallel_reactions[i]['Model Names'][x]).flux_expression /
                    parallel_reactions[i]['Model Directionality'][x] for x in
                    range(len(parallel_reactions[i]['Model Names']))),
                lb=parallel_reactions[i]['LB'], ub=parallel_reactions[i]['UB'])
            model.add_cons_vars(summed_reactions[parallel_reactions[i]['C13 Name']])

    constrained_model = model

    return(constrained_model)