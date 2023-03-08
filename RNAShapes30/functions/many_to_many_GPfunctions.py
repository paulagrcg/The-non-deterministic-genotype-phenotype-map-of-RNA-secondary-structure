#!/usr/bin/env python3
import numpy as np
from multiprocessing import Pool
import random
from copy import deepcopy
import RNA
from functools import partial

base_list = ['A', 'U', 'G', 'C']

def substitution_neighbours_letters(g, L):  
   return [''.join([base if seqindex!=pos else new_base for seqindex, base in enumerate(g)]) for pos in range(L) for new_base in base_list if not g[pos] == new_base]

def sequence_robustness_and_phipq(seq, GPfunction):
   """apply GPfunction to all sequences in the mutational neighbourhood and summarise
   neutral and non-neutral neighbours (genotype robustness and 
   concatenation of non-neutral shapes sererated by 'o' character"""
   structure, L = GPfunction(seq), len(seq)
   assert structure != '|'
   g_list = substitution_neighbours_letters(seq, L)
   assert len(g_list) == 3 * L
   p_neighbours = [GPfunction(g) for g in g_list]
   rho = p_neighbours.count(structure)/float(len(g_list))
   phi = [s for s in p_neighbours if s != structure]
   return structure, rho, 'o'+'o'.join(phi)

def get_evolvability_from_list(new_structures_list, structure_initial):
    assert '|' not in new_structures_list
    return len(set([structure for structure in new_structures_list.split('o') if structure != structure_initial ])) #and structure != '_'


def get_shape_from_random_sequence(repetition, L, GPfunction):
   """generate a random sequence of length L, 
      apply the GPfunction and return the sequence and GPfunction(sequence);
      repetition is an unused variable and introduced only so that pool.map can be used"""
   seq = ''.join([random.choice(['A', 'U', 'C', 'G']) for c in range(L)])
   return GPfunction(seq)
###############################################################################################

def robustness_many_to_many(seq, shape_distribution_function):
    print('robustness_many_to_many', seq, flush=True)
    nd_genotype_rob = 0.0
    initial_structure_distribution = shape_distribution_function(seq)
    for alternative_seq in substitution_neighbours_letters(seq, len(seq)):
        neighbour_structure_distribution = shape_distribution_function(alternative_seq)
        nd_genotype_rob += float(sum([Ps * neighbour_structure_distribution[s] if s in neighbour_structure_distribution else 0 for s, Ps in initial_structure_distribution.items()]))/(3 * len(seq))
    return nd_genotype_rob, max([P for P in initial_structure_distribution.values()])


def get_shape_distribution_from_random_sequence(repetition, L, shape_distribution_function):
   """generate a random sequence of length L, 
      apply the shape_distribution_function and return the sequence and GPfunction(sequence);
      repetition is an unused variable and introduced only so that pool.map can be used"""
   seq = ''.join([random.choice(['A', 'U', 'C', 'G']) for c in range(L)])
   return shape_distribution_function(seq)

def evolvability_many_to_many(seq, shape_distribution_function, list_all_structures):
    print('evolvability_many_to_many', seq, flush=True)
    nd_genotype_ev = 0.0
    seq_vs_distribution = {alternative_seq: shape_distribution_function(alternative_seq) for alternative_seq in substitution_neighbours_letters(seq, len(seq))}
    initial_structure_distribution = shape_distribution_function(seq)
    for structure, structureP in initial_structure_distribution.items():
           for alternative_struct in list_all_structures:
                if alternative_struct != structure:
                    nd_genotype_ev += initial_structure_distribution[structure] * (1- np.prod([1 - dict_value_or_zero(alternative_struct, alternative_distr) for alternative_distr in seq_vs_distribution.values()]))
    return nd_genotype_ev

def seq_data_forpheno_rob_nd(seq, shape_list, shape_distribution_function):
   print('phenotype robustness for', seq, flush=True)
   L = len(seq)
   distribution_seq = shape_distribution_function(seq)
   neighbour_vs_dist = {seq2: shape_distribution_function(seq2) for seq2 in substitution_neighbours_letters(seq, L)}
   norm = [dict_value_or_zero(shape, distribution_seq) * 3 * L for shape in shape_list]
   summand = [sum([dict_value_or_zero(shape, distribution_seq) * dict_value_or_zero(shape, ndist) for ndist in neighbour_vs_dist.values()]) for shape in shape_list]
   return summand, norm


def dict_value_or_zero(key_dict, dictionary):
  try:
    return dictionary[key_dict]
  except KeyError:
    return 0
###############################################################################################

"""
def frozen_version_of_map(iteration, all_seq_and_neighbours_vs_dist, seq_list):
   shape_list, rob_list, ev_list = [], [], []
   seq_vs_structure = {deepcopy(s): random.choices([struct for struct in d.keys()], weights=[d[struct] for struct in d.keys()])[0] for s, d in all_seq_and_neighbours_vs_dist.items()}
   for seq in seq_list:
      shape = seq_vs_structure[seq]
      rob = len([1 for s in substitution_neighbours_letters(seq, len(seq)) if seq_vs_structure[s] == shape])/float(3 * len(seq))
      evolv = len(set([seq_vs_structure[s] for s in substitution_neighbours_letters(seq, len(seq)) if seq_vs_structure[s] != shape]))
      shape_list.append(shape)
      rob_list.append(rob)
      ev_list.append(evolv)
   return  shape_list, rob_list, ev_list

def frozen_version_of_map_incl_folding(Boltzmann_draw_function, seq_list):
   par_function = partial(frozen_version_of_map_rob_evolv, Boltzmann_draw_function = Boltzmann_draw_function)
   with Pool(25) as p:
      pool_result = p.map(par_function, seq_list)
   return  zip(*pool_result)


def frozen_version_of_map_rob_evolv(seq, Boltzmann_draw_function):
   shape = Boltzmann_draw_function(seq)
   neighbour_shapes = [Boltzmann_draw_function(s) for s in substitution_neighbours_letters(seq, len(seq))]
   rob = len([1 for n in neighbour_shapes if n == shape])/float(3 * len(seq))
   evolv = len(set([n for n in neighbour_shapes if n != shape]))
   return shape, rob, evolv"""

def frozen_version_of_map_incl_folding_at_once(Boltzmann_draw_function, seq_list):
  L = len(seq_list[0])
  seq_list_incl_neighbours = [deepcopy(s) for s in seq_list] + [sn for s in seq_list for sn in substitution_neighbours_letters(s, L)]
  with Pool(25) as p:
    pool_result = p.map(Boltzmann_draw_function, seq_list_incl_neighbours)
  seq_vs_structure = {deepcopy(s): deepcopy(struct) for s, struct in zip(seq_list_incl_neighbours, pool_result)}
  del seq_list_incl_neighbours, pool_result
  shape_list = [seq_vs_structure[seq] for seq in seq_list]
  rob_list = [len([1 for sn in substitution_neighbours_letters(seq, L) if seq_vs_structure[sn] == seq_vs_structure[seq]])/float(3 * L) for seq in seq_list]
  evolv_list = [len(set([seq_vs_structure[sn] for sn in substitution_neighbours_letters(seq, L) if seq_vs_structure[sn] != seq_vs_structure[seq]])) for seq in seq_list]
  return shape_list, rob_list, evolv_list


def draw_from_Boltzmann_ensemble_full_dotbracket(seq, shape_and_prob_function):
  shape_vs_P = shape_and_prob_function(seq)
  shape_list = [s for s in shape_vs_P.keys()]
  return random.choices(shape_list, weights=[shape_vs_P[s] for s in shape_list], k=1)[0]

def get_shape_from_random_sequence_Boltzmann_draws(repetition, L, shape_and_prob_function, iterations):
   """generate a random sequence of length L, 
      apply the GPfunction and return the sequence and GPfunction(sequence);
      repetition is an unused variable and introduced only so that pool.map can be used"""
   seq = ''.join([random.choice(['A', 'U', 'C', 'G']) for c in range(L)])
   shape_vs_P = shape_and_prob_function(seq)
   shape_list = [s for s in shape_vs_P.keys()]
   return random.choices(shape_list, weights=[shape_vs_P[s] for s in shape_list], k=iterations)


