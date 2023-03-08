import numpy as np
import functions.Vienna_cgshape_probs as shapes
from functools import partial
from functions.rna_structural_functions import dotbracket_to_coarsegrained_for_level
###############################################################################################
# 'general' 
###############################################################################################
shape_level, range_kbT = 2, 15
K = 4 #alphabet size (4 for RNA), following notation by Schaper and Louis 
dangling_ends_option = 2 # corresponds to overdangle grammar
folding_criterion = 0.99 #this means that despite rounding errors all sequences will fold into most frequent structure in mfe case
L = 30
allow_isolated_bps = False
param_general = '_' + str(int(10*folding_criterion)) + '_' + str(dangling_ends_option) + 'SL' + str(shape_level) + 'kbT' + str(range_kbT)
###############################################################################################
# 'G-sampling parameters for large G-sample (frequencies)' 
###############################################################################################
sample_sizeGsample = 10**8 #10**5
parametersGsample = 'L'+str(L)+'_gsample'+str(int(np.log10(sample_sizeGsample))) + param_general 
minimum_number_found = 10 
###############################################################################################
# 'G-sampling parameters for small G-sample (robustness/evolv)' 
###############################################################################################
sample_sizeGsample_small = 10**6 #10**5
parametersGsample_small = 'L'+str(L)+'_gsample'+str(int(np.log10(sample_sizeGsample_small))) + param_general 
###############################################################################################
# 'sub-sampling parameters for small G-sample' 
###############################################################################################
subsample_ratio = 10
parametersGsample_small_subsample = 'L'+str(L)+'_gsample_subsample'+str(int(np.log10(sample_sizeGsample_small//subsample_ratio))) + param_general 
parametersGsample_small_divten = 'L'+str(L)+'_gsample_smaller'+str(int(np.log10(sample_sizeGsample_small//subsample_ratio))) + param_general 
parametersGsample_small_subsubsample = 'L'+str(L)+'_gsample_smaller_subsample'+str(int(np.log10(sample_sizeGsample_small//subsample_ratio**2))) + param_general
###############################################################################################
# 'smaller sample for average case' 
###############################################################################################
no_iterations = 500 #100
if sample_sizeGsample_small < 2 * 10**5:
   genotypic_filename_av_many_to_one = './data/genotype_info_av_many_to_one_gsample'+parametersGsample_small+'iterations'+str(no_iterations)+'.csv'
   prho_filename_av = './data/phenotype_robustness_av_many_to_one_gsample'+parametersGsample_small+'iterations'+str(no_iterations)+'.csv'
   genotypic_filename_av_many_to_one_subsample = './data/genotype_info_av_many_to_one_gsample'+parametersGsample_small_subsample+'iterations'+str(no_iterations)+'.csv'
   prho_filename_av_subsample = './data/phenotype_robustness_av_many_to_one_gsample'+parametersGsample_small_subsample+'iterations'+str(no_iterations)+'.csv'
else:
   genotypic_filename_av_many_to_one = './data/genotype_info_av_many_to_one_gsample'+parametersGsample_small_divten+'iterations'+str(no_iterations)+'.csv'
   prho_filename_av = './data/phenotype_robustness_av_many_to_one_gsample'+parametersGsample_small_divten+'iterations'+str(no_iterations)+'.csv'
   genotypic_filename_av_many_to_one_subsample = './data/genotype_info_av_many_to_one_gsample'+parametersGsample_small_subsubsample+'iterations'+str(no_iterations)+'.csv'
   prho_filename_av_subsample = './data/phenotype_robustness_av_many_to_one_gsample'+parametersGsample_small_subsubsample+'iterations'+str(no_iterations)+'.csv'
###############################################################################################
# functions 
###############################################################################################
GPfunction = partial(shapes.find_most_freq_shape, shape_level=shape_level, range_kbT=range_kbT, folding_criterion=folding_criterion, allow_isolated_bps=allow_isolated_bps, dangling_ends_option=dangling_ends_option)
cg_function = partial(dotbracket_to_coarsegrained_for_level, shape_level=shape_level)
shape_and_prob_function = partial(shapes.get_shapes_prob_subopt, shape_level=shape_level, range_kbT=range_kbT, allow_isolated_bps=allow_isolated_bps, dangling_ends_option=dangling_ends_option)
