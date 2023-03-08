#!/usr/bin/env python3
import numpy as np
import pandas as pd
from functools import partial
from os.path import isfile
from multiprocessing import Pool
import random
#from functions.many_to_many_GPfunctions import *
from functions.rna_structural_functions import generate_all_allowed_dotbracket, sequence_compatible_with_basepairs
import RNA
from collections import Counter
from itertools import product

kbT_RNA = RNA.exp_param().kT/1000.0 ## from https://github.com/ViennaRNA/ViennaRNA/issues/58
base_to_number={'A': 0, 'C': 1, 'U': 2, 'G': 3}
number_to_base = {n: b for b, n in base_to_number.items()}
K = 4


def substitution_neighbours_ind(g, L):  
   return [tuple([base if seqindex!=pos else new_base for seqindex, base in enumerate(g)]) for pos in range(L) for new_base in range(4) if not g[pos] == new_base]


def Boltzmann_dist(seq_tuple, structure_dotbracket_list, allow_isolated_bps):
   seq = ''.join([number_to_base[s] for s in seq_tuple])
   md = RNA.md()
   if not allow_isolated_bps:
      md.noLP = 1 #no isolated base pairs
   md.uniq_ML = 1
   a = RNA.fold_compound(seq, md)
   (mfe_structure, mfe) = a.mfe()
   a.exp_params_rescale(mfe)   
   structure_vs_energy = {s: a.eval_structure(s) for s in structure_dotbracket_list if sequence_compatible_with_basepairs(seq, s)}
   weight_list = {s: np.exp(np.array(e) * -1.0/kbT_RNA) for s, e in structure_vs_energy.items()}
   Z = np.sum([w for w in weight_list.values()])
   return {s: w/Z for s, w in weight_list.items()}

def Boltzmann_array(structure_dotbracket_list, allow_isolated_bps, Boltzmann_arrayfilename):
  if not isfile(Boltzmann_arrayfilename):
     Boltzmann_array = np.zeros(tuple([K, ] * L + [len(structure_dotbracket_list),]), dtype=np.single)
     progress = 0
     for g_index, e in np.ndenumerate(np.zeros(tuple([K, ] * L), dtype=np.half)):
        structure_vs_freq = Boltzmann_dist(g_index, structure_dotbracket_list, allow_isolated_bps)
        for structureindex, structure in enumerate(structure_dotbracket_list):
            if structure in structure_vs_freq:
               Boltzmann_array[tuple([g for g in g_index] + [structureindex,])] = structure_vs_freq[structure]
            else:
               Boltzmann_array[tuple([g for g in g_index] + [structureindex,])] = np.nan
        del structure_vs_freq
        progress += 1
        if progress % 10**4 == 0:
          print('finished', progress/4.0**L * 100,'% of Boltzmann array', flush=True)
     np.save(Boltzmann_arrayfilename, Boltzmann_array, allow_pickle=False)
  else:
     Boltzmann_array = np.load(Boltzmann_arrayfilename)
  return Boltzmann_array

def Boltzmann_from_Boltzmann_array(seq, Boltzmann_array, structure_dotbracket_list):
   seq_tuple = [base_to_number[s] for s in seq]
   return {s: Boltzmann_array[tuple(seq_tuple + [sindex])] for sindex, s in enumerate(structure_dotbracket_list) if not np.isnan(Boltzmann_array[tuple(seq_tuple + [sindex])])}

def mfe_freq_from_Boltzmann_array_seq_sample(L, num_shapes):
   shapeindex_vs_freq = {i: 0 for i in range(num_shapes)}
   for g, dummy in np.ndenumerate(np.zeros((K,)*L, dtype=np.half)):
      Boltzmann_list = [Boltzmann_array[tuple([x for x in g] + [shapeindex,])] for shapeindex in range(num_shapes)]
      assert abs(sum([B for B in Boltzmann_list if not np.isnan(B)]) - 1) < 0.01
      mfe_index = np.nanargmax(Boltzmann_list)
      shapeindex_vs_freq[mfe_index] += 1
   return shapeindex_vs_freq 
 
def pheno_rob_nd_from_Boltzmann_array(shapeindex, L):
   print('phenotype robustness for index', shapeindex, flush=True)
   rho, norm = 0.0, 0.0
   for g, dummy in np.ndenumerate(np.zeros((K,)*L, dtype=np.half)):
      Boltz = Boltzmann_array[tuple([x for x in g] + [shapeindex,]) ]
      if not np.isnan(Boltz):
        for genotype_ind2 in substitution_neighbours_ind(g, L):
           Boltz2 = Boltzmann_array[tuple([g for g in genotype_ind2] + [shapeindex,])]
           if not np.isnan(Boltz2):
              rho += Boltz * Boltz2/float(3 * L) 
        norm += Boltz
   return rho/norm 

  

def pheno_rob_nd_from_Boltzmann_array_seq_sample(shapeindex, L, seq_list):
   print('phenotype robustness for index', shapeindex, flush=True)
   rho, norm = 0, 0
   for seq in seq_list:
      g = tuple([base_to_number[s] for s in seq])
      Boltz = Boltzmann_array[tuple([x for x in g] + [shapeindex,]) ]
      if not np.isnan(Boltz):
        for genotype_ind2 in substitution_neighbours_ind(g, L):
           Boltz2 = Boltzmann_array[tuple([g for g in genotype_ind2] + [shapeindex,])]
           if not np.isnan(Boltz2):
              rho += Boltz * Boltz2/float(3 * L) 
        norm += Boltz
   assert norm > 0
   return rho/norm      

# def get_mfe_structure(seq, allow_isolated_bps):
#    md = RNA.md()
#    if not allow_isolated_bps:
#       md.noLP = 1 #no isolated base pairs
#    md.uniq_ML = 1
#    a = RNA.fold_compound(seq, md)
#    (mfe_structure, mfe) = a.mfe()
#    return mfe_structure
###############################################################################################
###############################################################################################
print('get all shapes')
###############################################################################################
###############################################################################################
allow_isolated_bps = True
L = 12
sample_sizeGsample_small = 10**4
shape_sample = generate_all_allowed_dotbracket(L, allow_isolated_bps=allow_isolated_bps)
shape_sample.append('.' * L)
filename_allstructures = './data/allstructuresL' + str(L) + 'csv'
df_structures = pd.DataFrame.from_dict({'structure': shape_sample})
df_structures.to_csv(filename_allstructures)
Boltzmann_arrayfilename = './data/RNA_Boltzmann'+str(len(shape_sample))+'_'+str(len(shape_sample[0]))+'.npy'
parametersGsample_small = 'L'+str(L)+'_gsample'+str(int(np.log10(sample_sizeGsample_small))) 

if isfile(Boltzmann_arrayfilename):
  Boltzmann_array = np.load(Boltzmann_arrayfilename)
else:
   Boltzmann_array = Boltzmann_array(shape_sample, allow_isolated_bps=allow_isolated_bps, Boltzmann_arrayfilename=Boltzmann_arrayfilename)
print('loaded Boltzmann aray', flush=True)
shape_distribution_function = partial(Boltzmann_from_Boltzmann_array, Boltzmann_array=Boltzmann_array, structure_dotbracket_list=shape_sample)
###############################################################################################
###############################################################################################
print('load data')
###############################################################################################
###############################################################################################

###############################################################################################
###############################################################################################
print('test structures that do not occur in Paulas data', flush=True)
###############################################################################################
###############################################################################################
for structure in ['(.....(...))' ,'((...).....)', '((...)(...))']:
    structureindex = shape_sample.index(structure)
    prob_compared_to_mfe = []
    for g, dummy in np.ndenumerate(np.zeros((K,)*L, dtype=np.half)):
      Boltz = Boltzmann_array[tuple([x for x in g] + [structureindex,]) ]
      if not np.isnan(Boltz):
         prob_compared_to_mfe.append(Boltz/np.nanmax([Boltzmann_array[tuple([x for x in g] + [structureindex2,]) ] for structureindex2 in range(len(shape_sample))]))
         if abs(kbT_RNA * np.log(prob_compared_to_mfe[-1])) < 15:
            print('structure', structure, 'seq', ''.join([number_to_base[s] for s in g]), 'dG', abs(kbT_RNA * np.log(prob_compared_to_mfe[-1])))
    deltaE_list = [abs(kbT_RNA * np.log(x)) for x in prob_compared_to_mfe]
    print('structure', structure)
    print('minimum deltaG', np.min(deltaE_list), 'typical deltaG', np.mean(deltaE_list), flush=True)


###############################################################################################
###############################################################################################
print('data for the many-to-one case: phenotypic frequency', flush=True)
###############################################################################################
###############################################################################################
#GPfunction = partial(get_mfe_structure, allow_isolated_bps=allow_isolated_bps)
N_filename_many_to_one = './data/fullstructures_genotype_info_many_to_one_neutral_sets' + str(L) + '.csv'
if not isfile(N_filename_many_to_one):
   #sequence_sample_list = [''.join(p) for p in product(['A', 'U', 'C', 'G'], repeat=L)]
   #with Pool(processes = 20) as p:
   #   pool_result = p.map(GPfunction, sequence_sample_list)
   structureindex_vs_f = mfe_freq_from_Boltzmann_array_seq_sample(L, len(shape_sample))
   structure_vs_freq = {shape_sample[shapeindex]: f for shapeindex, f in structureindex_vs_f.items()}
   df_genotypes = pd.DataFrame.from_dict({'structure': shape_sample, 
                                          'neutral set size': [structure_vs_freq[s] for s in shape_sample]})
   df_genotypes.to_csv(N_filename_many_to_one)
else:
   df_genotypes = pd.read_csv(N_filename_many_to_one)
   structure_vs_freq = {row['structure']: row['neutral set size'] for rowindex, row in df_genotypes.iterrows()}
   print('smallest non-zero neutral set size', min([n for n in df_genotypes['neutral set size'].tolist() if n > 0.5]))
###############################################################################################
###############################################################################################
print('data for the many-to-many case: phenotypic robustness', flush=True)
###############################################################################################
###############################################################################################
for sample in ['full', parametersGsample_small]:
  prho_filename_many_to_many = './data/fullstructures_phenotype_robustness_many_to_many'+sample+'.csv'
  if not isfile(prho_filename_many_to_many) and not sample == 'full':
      sequence_sample_list = [''.join([random.choice(['A', 'U', 'C', 'G']) for c in range(L)]) for i in range(sample_sizeGsample_small)]
      robustness_function_all_param = partial(pheno_rob_nd_from_Boltzmann_array_seq_sample, L=L, seq_list = sequence_sample_list)
      with Pool(processes = 20) as p:
         shape_rob_list = p.map(robustness_function_all_param, np.arange(len(shape_sample)))
      df_phenotypes_nd_rob = pd.DataFrame.from_dict({'structure': shape_sample, 
                                                    'phenotype robustness': shape_rob_list})
      df_phenotypes_nd_rob.to_csv(prho_filename_many_to_many) 
  if not isfile(prho_filename_many_to_many):
      print('find phenotype robustness based on saved array')
      robustness_function_all_param = partial(pheno_rob_nd_from_Boltzmann_array, L=L)
      with Pool(processes = 20) as p:
         shape_rob_list = p.map(robustness_function_all_param, np.arange(len(shape_sample)))
      df_phenotypes_nd_rob = pd.DataFrame.from_dict({'structure': shape_sample, 
                                                    'phenotype robustness': shape_rob_list})
      df_phenotypes_nd_rob.to_csv(prho_filename_many_to_many) 
###############################################################################################
###############################################################################################
print('quick plot', flush=True)
###############################################################################################
###############################################################################################
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
df_phenotypes_nd_rob = pd.read_csv('./data/fullstructures_phenotype_robustness_many_to_many'+'full'+'.csv')
df_phenotypes_nd_rob_sample = pd.read_csv('./data/fullstructures_phenotype_robustness_many_to_many'+parametersGsample_small+'.csv')
#sampling_quality = [(structure_vs_freq[shape]/4**L)/(sample_sizeGsample_small/4**L) for shape in df_phenotypes_nd_rob_sample['structure'].tolist()]
f, ax = plt.subplots(figsize=(4,3))
s = ax.scatter(df_phenotypes_nd_rob['phenotype robustness'].tolist(), df_phenotypes_nd_rob_sample['phenotype robustness'].tolist(),
           marker ='x', s=6, zorder=1)
#print({structure: freq/4**L for structure, freq in structure_vs_freq.items()})
#cb = f.colorbar(s, ax=ax)
#cb.set_label('log10 fraction of sequences  sampled/\nstructural frequency')
ax.plot([0,1], [0, 1], c='k', zorder=-2, lw=0.5)
ax.set_ylabel('non-deterministic\nphenotype robustness '+r'$\widetilde{\rho_p}$' + '\nfrom sequence sample')
ax.set_xlabel('non-deterministic\nphenotype robustness '+r'$\widetilde{\rho_p}$' + '\nexact data')
ax.set_title('sample size:' + str(sample_sizeGsample_small))
f.tight_layout()
f.savefig('./plots/compare_metrics'+parametersGsample_small+'phrho_gsample.png', dpi=300, bbox_inches='tight')
plt.close('all')
del f, ax
###############################################################################################
###############################################################################################
print('quick plot', flush=True)
###############################################################################################
###############################################################################################
phenotypic_frequency_nd = './data/fullstructures_phenotype_freq_many_to_many'+'full'+'.csv'
if not isfile(phenotypic_frequency_nd):
  pheno_freq = [np.nansum(Boltzmann_array[:, :, :, :, :, :, :, :, :, :, :, :, i])/4**L for i in range(len(shape_sample))]
  df_freq_nd = pd.DataFrame.from_dict({'shape': shape_sample, 'phenotype freq': pheno_freq})
  df_freq_nd.to_csv(phenotypic_frequency_nd)
else:
  df_freq_nd = pd.read_csv(phenotypic_frequency_nd)
f, ax = plt.subplots(figsize=(4,3))
ax.set_xscale('log')
ax.set_xlim(1/4**L, 1)
s = ax.scatter(df_freq_nd['phenotype freq'].tolist(), df_phenotypes_nd_rob['phenotype robustness'].tolist(),
           marker ='x', s=6, zorder=1)
ax.plot(np.power(10, np.linspace(np.log10(1/4**L),0, 10**4)), np.power(10, np.linspace(np.log10(1/4**L),0, 10**4)), 
           c='k', zorder=-2)

ax.set_ylabel('non-deterministic\nphenotype robustness'+r'$\widetilde{\rho_p}$')
ax.set_xlabel('non-deterministic\nphenotype frequency'+r'$\widetilde{f_p}$' )
f.tight_layout()
f.savefig('./plots/freq_rob'+parametersGsample_small+'phrho_gsample.png', dpi=300, bbox_inches='tight')
plt.close('all')
del f, ax



