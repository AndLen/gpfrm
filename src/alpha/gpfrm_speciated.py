from itertools import starmap

from deap import base
from deap import creator
from numba import njit

import gptools.weighted_generators as wg
from alpha import frm_speciated_ea as ase
from alpha.eval_frm import st_pearsons
from alpha.gp_design import get_pset_weights
from alpha.rundata_frm_sp import rd
from gptools.ParallelToolbox import ParallelToolbox
from gptools.gp_util import *
from gptools.multitree import *


# https://github.com/erikbern/ann-benchmarks
# https://github.com/nmslib/hnswlib
# https://github.com/nmslib/nmslib

def main():
    pop = toolbox.population(n=rd.pop_size)
    stats_cost = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_raw_cost = tools.Statistics(lambda ind: ind.raw_cost)
    stats_parsimony = tools.Statistics(lambda ind: ind.parsimony)
    mstats = tools.MultiStatistics(cost=stats_cost, raw_cost=stats_raw_cost, size=stats_parsimony)
    mstats.register("min", np.min, axis=0)
    mstats.register("median", np.median, axis=0)
    mstats.register("max", np.max, axis=0)
    mstats['basic'] = tools.Statistics()
    eval_func = partial(st_pearsons, rd.data_t, rd.matching_features_array,rd.matching_features_array_len)
    assert rd.max_trees == 1, 'Only single-tree individuals supported for Speciation GP-frm.'
    pop, logbook, species_seeds = ase.ea(pop, toolbox, rd.cxpb, rd.mutpb, rd.gens, eval_func=eval_func, stats=mstats,
                                         verbose=True)
    return pop, mstats, [], logbook


def make_ind(toolbox, creator, num_trees):
    return creator.Individual([toolbox.tree() for _ in range(num_trees)])


def get_extra_args():
    # algorithm specific
    arg_list = arg_parse_helper('-alpha', '--alpha', help='Alpha value for parsimony weighting', type=float,
                                dest='alpha')
    arg_parse_helper('-sp', '--species', help='Number of species to use', type=int, dest='max_species',
                     arg_list=arg_list)
    return arg_list


@njit(parallel=True)
def fast_pairwise_pearsons(data_t, identical_threshold):
    num_of = data_t.shape[0]
    abs_pearsons = np.zeros((num_of, num_of))
    matching_features = []
    for of in range(num_of):
        matching_feats = []
        for of2 in range(num_of):
            # could do half the square since symmetrical, but only do this once anyway
            np_abs = np.abs(np.corrcoef(data_t[of], data_t[of2])[0, 1])
            abs_pearsons[of][of2] = np_abs
            if np_abs > identical_threshold:
                matching_feats.append(of2)
        matching_features.append(matching_feats)
    return abs_pearsons, matching_features


# https://stackoverflow.com/questions/53314071/turn-off-list-reflection-in-numba
def make_2D_array(lis):
    """Funciton to get 2D array from a list of lists
    """
    n = len(lis)
    lengths = np.array([len(x) for x in lis])
    max_len = np.max(lengths)
    arr = np.zeros((n, max_len))

    for i in range(n):
        arr[i, :lengths[i]] = lis[i]
    return arr, lengths


if __name__ == "__main__":
    arg_list = get_extra_args()
    init_data(rd, additional_arguments=arg_list)

    pset, weights = get_pset_weights(rd.num_features, rd)
    rd.pset = pset
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * rd.nobj)
    creator.create("Individual", list, fitness=creator.FitnessMin, pset=pset)

    toolbox = ParallelToolbox()  #

    toolbox.register("expr", wg.w_genHalfAndHalf, pset=pset, weighted_terms=weights, min_=0, max_=rd.max_depth)
    toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
    toolbox.register("individual", make_ind, toolbox, creator, rd.max_trees)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("try_cache", try_cache, rd)

    toolbox.register("evaluate", st_pearsons, rd, rd.data_t, toolbox)
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", lim_xmate_aic, max_height=rd.max_depth)

    toolbox.register("expr_mut", wg.w_genFull, weighted_terms=weights, min_=0, max_=rd.max_depth)
    toolbox.register("mutate", lim_xmut, expr=toolbox.expr_mut, max_height=rd.max_depth)
    toolbox.register("starmap", starmap)
    assert math.isclose(rd.cxpb + rd.mutpb, 1), "Probabilities of operators should sum to ~1."

    print(rd)

    # find similar source features.
    num_of = rd.data_t.shape[0]
    identical_threshold = 0.95

    rd.abs_pearsons, rd.matching_features = fast_pairwise_pearsons(rd.data_t, identical_threshold)
    rd.matching_features_array, rd.matching_features_array_len = make_2D_array(rd.matching_features)
    print(rd.matching_features)
    # probably the initializer stuff...?

    pop, stats, hof, logbook = main()

    final_output(hof, toolbox, logbook, pop, rd, classify=True, plot_curve=True)
