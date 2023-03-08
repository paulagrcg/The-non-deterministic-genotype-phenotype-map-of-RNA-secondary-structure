#!/usr/bin/env python3
import numpy as np
import pandas as pd
from functools import partial
from os.path import isfile
from multiprocessing import Pool
import random
from copy import deepcopy
import parameters as param
from functions.many_to_many_GPfunctions import *
from functions.rna_structural_functions import generate_all_allowed_dotbracket

Boltzmann_draw_function = partial(draw_from_Boltzmann_ensemble_full_dotbracket, shape_and_prob_function=param.shape_and_prob_function)
###############################################################################################
###############################################################################################
print('get all shapes')
###############################################################################################
###############################################################################################
filename_shape_list = './data/allshapesL'+str(param.L)+'.csv'
if isfile(filename_shape_list):
    shape_sample = pd.read_csv(filename_shape_list)['shape'].tolist()
else:
    full_secondary_str_list = generate_all_allowed_dotbracket(param.L, allow_isolated_bps=param.allow_isolated_bps)
    full_shapes_list = [param.cg_function(s) for s in full_secondary_str_list] + ['_']
    shape_sample = sorted(list(set(full_shapes_list)), key=len)
    df_shape_smple = pd.DataFrame.from_dict({'shape': shape_sample}) 
    df_shape_smple.to_csv(filename_shape_list)
###############################################################################################
###############################################################################################
###############################################################################################
### many-to-one
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
print('data for the many-to-one case: genotypic robustness and evolvability')
###############################################################################################
###############################################################################################
genotypic_filename_many_to_one = './data/genotype_info_many_to_one_gsample'+param.parametersGsample_small+'.csv'

if not isfile(genotypic_filename_many_to_one):
    sequence_sample_list = [''.join([random.choice(['A', 'U', 'C', 'G']) for c in range(param.L)]) for i in range(param.sample_sizeGsample_small)]
    function_for_parallelised_calc = partial(sequence_robustness_and_phipq, GPfunction=param.GPfunction)
    with Pool(processes = 20) as p:
       pool_result = p.map(function_for_parallelised_calc, sequence_sample_list)
    shape_list_sample, rho_listS, phi_listS = zip(*pool_result)
    df_genotypes = pd.DataFrame.from_dict({'structure': shape_list_sample, 
                                             'sequence': sequence_sample_list,
                                             'genotype robustness': rho_listS,
                                             'genotype evolvability': [get_evolvability_from_list(phi, shape) for phi, shape in zip(phi_listS, shape_list_sample)]})
    df_genotypes.to_csv(genotypic_filename_many_to_one)
else:
    df_genotypes = pd.read_csv(genotypic_filename_many_to_one)
    sequence_sample_list = df_genotypes['sequence'].tolist()
###############################################################################################
###############################################################################################
print('same for subsample')
###############################################################################################
###############################################################################################
genotypic_filename_many_to_one_subsample = './data/genotype_info_many_to_one_gsample'+param.parametersGsample_small_subsample +'.csv'
if not isfile(genotypic_filename_many_to_one_subsample):
    sequence_sample_list_subsample = deepcopy(sequence_sample_list[:len(sequence_sample_list)//param.subsample_ratio])
    function_for_parallelised_calc = partial(sequence_robustness_and_phipq, GPfunction=param.GPfunction)
    with Pool(processes = 20) as p:
       pool_result = p.map(function_for_parallelised_calc, sequence_sample_list_subsample)
    shape_list_sample, rho_listS, phi_listS = zip(*pool_result)
    df_genotypes_subsample = pd.DataFrame.from_dict({'structure': shape_list_sample, 
                                             'sequence': sequence_sample_list_subsample,
                                             'genotype robustness': rho_listS,
                                             'genotype evolvability': [get_evolvability_from_list(phi, shape) for phi, shape in zip(phi_listS, shape_list_sample)]})
    df_genotypes_subsample.to_csv(genotypic_filename_many_to_one_subsample)

###############################################################################################
###############################################################################################
print('data for the many-to-one case: extract phenotypic robustness')
###############################################################################################
###############################################################################################
for sample_params in [param.parametersGsample_small_subsample, param.parametersGsample_small]:
  df_genotypes_for_prho = pd.read_csv('./data/genotype_info_many_to_one_gsample'+sample_params+'.csv')
  phenotypic_filename_many_to_one = './data/phenotype_robustness_many_to_one_gsample'+sample_params+'.csv'
  if not isfile(phenotypic_filename_many_to_one):
    shape_vs_rho_values = {s: [] for s in set(df_genotypes_for_prho['structure'].tolist())}
    for rowindex, row in df_genotypes_for_prho.iterrows():
      shape_vs_rho_values[row['structure']].append(row['genotype robustness'])
    df_phenotypesD_rob = pd.DataFrame.from_dict({'structure': [s for s in shape_vs_rho_values.keys()], 
                                                'phenotype robustness': [np.mean(shape_vs_rho_values[s]) for s in shape_vs_rho_values.keys()]})
    df_phenotypesD_rob.to_csv(phenotypic_filename_many_to_one)  
    
###############################################################################################
###############################################################################################
###############################################################################################
### averages over many maps drawn from Boltzmann distribution
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
print('data for the average many-to-one case')
###############################################################################################
###############################################################################################

for (genotypic_filename_av_many_to_one, prho_filename_av) in [(param.genotypic_filename_av_many_to_one_subsample, param.prho_filename_av_subsample), (param.genotypic_filename_av_many_to_one, param.prho_filename_av)]:
  print(genotypic_filename_av_many_to_one, prho_filename_av)
  if prho_filename_av == param.prho_filename_av and param.sample_sizeGsample_small < 2 * 10**5: # use a smaller sample for average case
    sequence_sample_given_sample_param = deepcopy(sequence_sample_list[:])
  elif (prho_filename_av == param.prho_filename_av and param.sample_sizeGsample_small >= 2 * 10**5) or (prho_filename_av == param.prho_filename_av_subsample and param.sample_sizeGsample_small < 2 * 10**5):
    sequence_sample_given_sample_param = deepcopy(sequence_sample_list[:len(sequence_sample_list)//param.subsample_ratio])
  else:
    sequence_sample_given_sample_param = deepcopy(sequence_sample_list[:len(sequence_sample_list)//(param.subsample_ratio**2)])
  if not isfile(genotypic_filename_av_many_to_one) or not isfile(prho_filename_av):
      geno_vs_rob, geno_vs_evolv, iteration_vs_shape_vs_rob, shapes_with_data = {s: [] for s in sequence_sample_given_sample_param}, {s: [] for s in sequence_sample_given_sample_param}, {i: {} for i in range(param.no_iterations)}, set([])
      for iteration in range(param.no_iterations):
         print('iteration', iteration, 'sample size', len(sequence_sample_given_sample_param), flush=True)
         #shape_list, rob_list, evolv_list = frozen_version_of_map_incl_folding(Boltzmann_draw_function, sequence_sample_given_sample_param)
         shape_list, rob_list, evolv_list =  frozen_version_of_map_incl_folding_at_once(Boltzmann_draw_function, sequence_sample_given_sample_param)
         for r, e, s, shape in zip(rob_list, evolv_list, sequence_sample_given_sample_param, shape_list):
           geno_vs_rob[s].append(r)
           geno_vs_evolv[s].append(e)
           try:
             iteration_vs_shape_vs_rob[iteration][shape].append(r)
           except KeyError:
             iteration_vs_shape_vs_rob[iteration][shape] = [r,]   
           shapes_with_data.add(shape)    
      df_genotypes_av = pd.DataFrame.from_dict({'sequence': [s for s in sequence_sample_given_sample_param],
                                               'genotype robustness': [np.mean(geno_vs_rob[s]) for s in sequence_sample_given_sample_param],
                                               'genotype evolvability': [np.mean(geno_vs_evolv[s]) for s in sequence_sample_given_sample_param]})
      df_genotypes_av.to_csv(genotypic_filename_av_many_to_one)
      ###
      shapes_with_data = list(shapes_with_data)
      df_phenotypes_av_rob = pd.DataFrame.from_dict({'structure': shapes_with_data, 
                                                    'phenotype robustness': [np.mean([np.mean(iteration_vs_shape_vs_rob[iteration][shape]) for iteration in range(param.no_iterations) if shape in iteration_vs_shape_vs_rob[iteration]]) for shape in shapes_with_data]})
      df_phenotypes_av_rob.to_csv(prho_filename_av)  
  

###############################################################################################
###############################################################################################
###############################################################################################
### many-to-many
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
print('data for the many-to-many case: genotypic data', flush=True)
###############################################################################################
###############################################################################################
genotypic_filename_many_to_many = './data/genotype_info_many_to_many_gsample'+param.parametersGsample_small+'.csv'
if not isfile(genotypic_filename_many_to_many):
    evolvability_function = partial(evolvability_many_to_many, shape_distribution_function=param.shape_and_prob_function, list_all_structures=shape_sample)
    with Pool(processes = 20) as p:
       evolvability_genotypes = p.map(evolvability_function, sequence_sample_list[:])
    robustness_function = partial(robustness_many_to_many, shape_distribution_function=param.shape_and_prob_function)
    with Pool(processes = 20) as p:
       pool_result = p.map(robustness_function, sequence_sample_list)
    robustness_genotypes, Boltz_freq_mfe = zip(*pool_result)
    df_genotypes_nd = pd.DataFrame.from_dict({'sequence': sequence_sample_list,
                                             'genotype robustness': robustness_genotypes,
                                             'genotype evolvability': evolvability_genotypes,
                                             'highest Boltzmann freq.': Boltz_freq_mfe})
    df_genotypes_nd.to_csv(genotypic_filename_many_to_many)
else:
    df_genotypes_nd = pd.read_csv(genotypic_filename_many_to_many)



###############################################################################################
###############################################################################################
print('data for the many-to-many case: phenotypic data', flush=True)
###############################################################################################
###############################################################################################
for sample_params in [param.parametersGsample_small, param.parametersGsample_small_subsample]:
  prho_filename_many_to_many = './data/phenotype_robustness_many_to_many_gsample'+sample_params+'.csv'
  if not isfile(prho_filename_many_to_many):
      pheno_rob_function = partial(seq_data_forpheno_rob_nd, shape_list=shape_sample, 
                                   shape_distribution_function=param.shape_and_prob_function)   
      if sample_params == param.parametersGsample_small:
         with Pool(processes = 20) as p:
            shape_rob_list = p.map(pheno_rob_function, sequence_sample_list[:])
      else:
        smaller_seq_sample = sequence_sample_list[:len(sequence_sample_list)//param.subsample_ratio]
        print('use smaller seq sample', len(smaller_seq_sample))
        with Pool(processes = 20) as p:
            shape_rob_list = p.map(pheno_rob_function, smaller_seq_sample)        
      ###
      df_phenotypes_nd_rob = pd.DataFrame.from_dict({'structure': shape_sample, 
                                                    'phenotype robustness': [sum([l[0][i] for l in shape_rob_list])/sum([l[1][i] for l in shape_rob_list]) if sum([l[1][i] for l in shape_rob_list]) > 0 else np.nan for i, shape in enumerate(shape_sample)]})
      df_phenotypes_nd_rob.to_csv(prho_filename_many_to_many)  
      print('find frequencies through g-sampling for non-deterministic case', flush=True)
###############################################################################################
###############################################################################################
print('quick plot', flush=True)
###############################################################################################
###############################################################################################
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
df_phenotypes_nd_rob = pd.read_csv('./data/phenotype_robustness_many_to_many_gsample'+param.parametersGsample_small+'.csv')
structure_vs_pheno_rob_nd = {row['structure']: row['phenotype robustness'] for rowindex, row in df_phenotypes_nd_rob.iterrows()}
#
df_phenotypes_av_rob = pd.read_csv(param.prho_filename_av)
structure_vs_pheno_rob_av = {row['structure']: row['phenotype robustness'] for rowindex, row in df_phenotypes_av_rob.iterrows()}
#
df_phenotypesD_rob = pd.read_csv('./data/phenotype_robustness_many_to_one_gsample'+param.parametersGsample_small +'.csv')
structure_vs_pheno_robD = {row['structure']: row['phenotype robustness'] for rowindex, row in df_phenotypesD_rob.iterrows()}
###
f, ax = plt.subplots(ncols=7, figsize=(23,3))
ax[0].scatter(df_genotypes['genotype robustness'].tolist(), df_genotypes_nd['genotype robustness'].tolist(), marker ='x', s=2, alpha=0.2)
ax[0].plot([0,1], [0, 1], c='k', zorder=-2)
ax[0].set_xlabel(r'genotype robustness $\rho_g$')
ax[0].set_ylabel(r'genotype robustness $\widetilde{\rho_g}$' + '\nnon-deterministic')
######
ax[1].scatter(df_genotypes['genotype evolvability'].tolist(), df_genotypes_nd['genotype evolvability'].tolist(), marker ='x', s=2, alpha=0.2)
ax[1].plot([0,max(df_genotypes['genotype evolvability'].tolist() + df_genotypes_nd['genotype evolvability'].tolist()) * 1.1], 
             [0,max(df_genotypes['genotype evolvability'].tolist() + df_genotypes_nd['genotype evolvability'].tolist()) * 1.1], c='k', zorder=-2)

ax[1].set_xlabel(r'genotype evolvability $e_g$')
ax[1].set_ylabel(r'genotype evolvability $\widetilde{e_g}$' + '\nnon-deterministic')
#####
df_genotypes_av = pd.read_csv(param.genotypic_filename_av_many_to_one)
data_points_av = len(df_genotypes_av['genotype robustness'].tolist())
ax[2].scatter(df_genotypes_nd['genotype robustness'].tolist()[:data_points_av], df_genotypes_av['genotype robustness'].tolist()[:data_points_av], marker ='x', s=2, alpha=0.2)
ax[2].plot([0,1], [0, 1], c='k', zorder=-2)
ax[2].set_xlabel(r'genotype robustness $\widetilde{\rho_g}$' + '\nnon-deterministic')
ax[2].set_ylabel(r'genotype robustness $\rho_g$' + '\naverage deterministic')
######
ax[3].scatter(df_genotypes_nd['genotype evolvability'].tolist()[:data_points_av], df_genotypes_av['genotype evolvability'].tolist()[:data_points_av], marker ='x', s=2, alpha=0.2)
ax[3].plot([0,max(df_genotypes_nd['genotype evolvability'].tolist() + df_genotypes_av['genotype evolvability'].tolist()) * 1.1], 
             [0,max(df_genotypes_nd['genotype evolvability'].tolist() + df_genotypes_av['genotype evolvability'].tolist()) * 1.1], c='k', zorder=-2)

ax[3].set_xlabel(r'genotype evolvability $\widetilde{e_g}$' + '\nnon-deterministic')
ax[3].set_ylabel(r'genotype evolvability $e_g$' + '\naverage deterministic')
######
ax[4].scatter([structure_vs_pheno_rob_nd[s] if s in structure_vs_pheno_rob_nd else 0 for s in shape_sample], 
              [structure_vs_pheno_rob_av[s] if s in structure_vs_pheno_rob_av else 0 for s in shape_sample], 
              marker ='x', s=7, alpha=0.8, zorder=1)
ax[4].plot([0,1], [0,1], c='k', zorder=-2, lw=0.5)

ax[4].set_xlabel(r'phenotype robustness $\widetilde{\rho_p}$' + '\nnon-deterministic')
ax[4].set_ylabel(r'phenotype robustness $\rho_p$' + '\naverage deterministic')
######
ax[5].scatter([structure_vs_pheno_rob_nd[s] if s in structure_vs_pheno_rob_nd else 0 for s in shape_sample], 
              [structure_vs_pheno_robD[s] if s in structure_vs_pheno_robD else 0 for s in shape_sample], 
              marker ='x', s=7, alpha=0.8, zorder=1)
ax[5].plot([0,1], [0,1], c='k', zorder=-2, lw=0.5)

ax[5].set_xlabel(r'phenotype robustness $\widetilde{\rho_p}$' + '\nnon-deterministic')
ax[5].set_ylabel(r'phenotype robustness $\rho_p$' + '\ndeterministic')
######
ax[6].scatter(df_genotypes['genotype robustness'].tolist(), df_genotypes_nd['highest Boltzmann freq.'].tolist(), marker ='x', s=2, alpha=0.2)
ax[6].plot([0,1], [0, 1], c='k', zorder=-2)
ax[6].set_xlabel(r'genotype robustness $\rho_g$')
ax[6].set_ylabel('Boltzmann frequency\n'+r'of mfe structure $p_g$')
f.tight_layout()
f.savefig('./plots/compare_metrics'+param.parametersGsample_small+'_gsample.png', dpi=300, bbox_inches='tight')
plt.close('all')
del f, ax
###############################################################################################
###############################################################################################
print('quick plot - effect of subsampling on phenotypic quantities', flush=True)
###############################################################################################
###############################################################################################
df_phenotypes_nd_rob_subsample = pd.read_csv('./data/phenotype_robustness_many_to_many_gsample'+param.parametersGsample_small_subsample+'.csv')
structure_vs_pheno_rob_nd_subsample = {row['structure']: row['phenotype robustness'] for rowindex, row in df_phenotypes_nd_rob_subsample.iterrows()}
##
df_phenotypes_av_rob_subsample = pd.read_csv(param.prho_filename_av_subsample)
structure_vs_pheno_rob_av_subsample = {row['structure']: row['phenotype robustness'] for rowindex, row in df_phenotypes_av_rob_subsample.iterrows()}
##
df_phenotypes_robD_subsample = pd.read_csv('./data/phenotype_robustness_many_to_one_gsample'+param.parametersGsample_small_subsample +'.csv')
structure_vs_pheno_robD_subsample = {row['structure']: row['phenotype robustness'] for rowindex, row in df_phenotypes_robD_subsample.iterrows()}
##
f, ax = plt.subplots(ncols=3, figsize=(9,2.5))
ax[0].scatter([structure_vs_pheno_rob_nd[s] for s in structure_vs_pheno_rob_nd], 
           [structure_vs_pheno_rob_nd_subsample[s] if s in structure_vs_pheno_rob_nd_subsample else 0 for s in structure_vs_pheno_rob_nd], 
           marker ='x', s=7, alpha=0.8, zorder=1)

ax[0].plot([0,1], [0, 1], c='k', zorder=-2, lw=0.5)
ax[0].set_title(r'non-deterministic map')
ax[0].set_xlabel(r'phenotype robustness'+ '\nbased on full sample')
ax[0].set_ylabel(r'phenotype robustness' + '\nbased on sub-sample')
###
ax[1].scatter([structure_vs_pheno_rob_av[s] for s in structure_vs_pheno_rob_av], 
           [structure_vs_pheno_rob_av_subsample[s] if s in structure_vs_pheno_rob_av_subsample else 0 for s in structure_vs_pheno_rob_av], 
           marker ='x', s=7, alpha=0.8, zorder=1)

ax[1].plot([0,1], [0, 1], c='k', zorder=-2, lw=0.5)
ax[1].set_title(r'average deterministic map')
ax[1].set_xlabel(r'phenotype robustness'+ '\nbased on full sample')
ax[1].set_ylabel(r'phenotype robustness' + '\nbased on sub-sample')
###
ax[2].scatter([structure_vs_pheno_robD[s] for s in structure_vs_pheno_robD], 
           [structure_vs_pheno_robD_subsample[s] if s in structure_vs_pheno_robD_subsample else 0 for s in structure_vs_pheno_robD], 
           marker ='x', s=7, alpha=0.8, zorder=1)

ax[2].plot([0,1], [0, 1], c='k', zorder=-2, lw=0.5)
ax[2].set_title(r'deterministic map')
ax[2].set_xlabel(r'phenotype robustness'+ '\nbased on full sample')
ax[2].set_ylabel(r'phenotype robustness' + '\nbased on sub-sample')
for i in range(3):
  ax[i].annotate('ABCDEF'[i], xy=(0.04, 0.86), xycoords='axes fraction', fontweight='bold') 
f.tight_layout()
f.savefig('./plots/subsampling_effect'+param.parametersGsample_small+'_probustness.png', dpi=300, bbox_inches='tight')
plt.close('all')
del f, ax
