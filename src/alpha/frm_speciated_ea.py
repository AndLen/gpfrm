import math
import random
from copy import deepcopy
from operator import attrgetter

from deap import tools

from alpha.rundata_frm_sp import rd
from gptools.array_wrapper import ArrayWrapper
from gptools.gp_util import output_ind, check_uniqueness_str, try_cache, get_arrays_cached, output_inds_simple, draw_ind

LOG_GENS = 100  # 0


def parsimony_pop(population):
    for ind in population:
        assert len(ind.fitness.values) == 1
        if not hasattr(ind, 'parsimony'):
            ind.raw_cost = ind.fitness.values[0]
            ind.parsimony = len(ind[0])
            ind.parsimony_weighted = rd.alpha * ind.parsimony
            ind.fitness.values = ind.raw_cost + ind.parsimony_weighted,


def ea(population, toolbox, cxpb, mutpb, ngen, eval_func, stats=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = (stats.fields if stats else [])

    for gen in range(1, ngen + 1):
        evaled = eval_population_cached_simple(population, toolbox, eval_func, rd)
        parsimony_pop(population)
        species = speciate(population)
        species_seeds = [species[x]['seed'] for x in species]
        record = stats.compile(species_seeds) if stats else {}
        if gen % LOG_GENS == 0:
            clean_up = False  # gen % 100 == 0
            for sp in species:
                suffix = "-f{}-{}N-{}".format(sp, len(species[sp]['species']), gen)
                seed_ = species[sp]['seed']
                output_ind(seed_, toolbox, rd, suffix,
                           del_old=clean_up, aug_file=False, columns=['F{}'.format(sp)],
                           fitnesses=[seed_.fitness.values[0], seed_.raw_cost, seed_.parsimony])
                draw_ind(seed_, toolbox, rd, eval_func, suffix)
                # so we only do it the first time!
                clean_up = False

            output_inds_simple(species_seeds, toolbox, rd, suffix="-{}-{}"
                               .format(len(species), gen), columns=['F{}'.format(x) for x in species])

        candidate_population = breed_offspring_speciated(cxpb, mutpb, species, toolbox, len(population))
        # replace the invalid solutions.
        pop_deficit = len(population) - len(candidate_population)
        candidate_population.extend(toolbox.population(n=pop_deficit))
        record['basic'] = {'gen': gen, 'nevals': len(evaled), 'nspecies': len(species), 'ninvalid': pop_deficit}

        logbook.record(**record)

        if verbose:
            print(logbook.stream)

        assert len(candidate_population) == len(population), 'Lost some population somewhere'
        population = candidate_population

    # one last time.
    evaled = eval_population_cached_simple(population, toolbox, eval_func, rd)
    parsimony_pop(population)

    species = speciate(population)
    species_seeds = [species[x]['seed'] for x in species]
    record = stats.compile(species_seeds) if stats else {}
    record['basic'] = {'gen': gen, 'nevals': len(evaled), 'nspecies': len(species), 'ninvalid': -1}

    logbook.record(**record)
    if verbose:
        print(logbook.stream)

    return population, logbook, species_seeds


def breed_offspring_speciated(cxpb, mutpb, species, toolbox, popsize):
    each_species_size = popsize // (len(species) + 1)
    candidate_population = []
    for sp in species:
        sp_seed = species[sp]['seed']
        sp_inds = species[sp]['species']
        sp_parents = toolbox.select(deepcopy(sp_inds), each_species_size - 1)

        # Vary the pool of individuals
        if len(sp_parents) > 0:
            sp_offspring = varOrUniqueBatched(sp_parents, toolbox, len(sp_parents), cxpb, mutpb)
            candidate_population.extend(sp_offspring)

        # species elitism. could probably make this "smarter" later.
        candidate_population.append(sp_seed)
    return candidate_population


def speciate(population):
    sorted_pop = sorted(population, key=attrgetter("fitness"), reverse=True)
    species = {}

    for next_ind in sorted_pop:
        # since an array of [0] evaluates to False.
        if next_ind.closest_features is not None:
            icf = next_ind.closest_features[0][0]
            if icf in species:
                species[icf]['species'].append(next_ind)
            # Need to have
            elif len(species) < rd.max_species:
                species[icf] = {'seed': next_ind,
                                'species': [next_ind]}
            else:
                i = 0
                while icf not in species:
                    i += 1
                    icf = next_ind.closest_features[0][i]
                # print(i)
                species[icf]['species'].append(next_ind)
        else:
            assert math.isinf(next_ind.fitness.values[0]), 'No closest feature, but is a valid individual.'
    return species


def varOrUniqueBatched(population, toolbox, lambda_, cxpb, mutpb):
    assert (cxpb + mutpb) == 1.0, (
        "The sum of the crossover and mutation probabilities must be equal to 1.0.")

    offspring = []
    no_change = 0
    while len(offspring) < lambda_ and no_change < 10:
        # how many pairs of candidates do we make?
        candidates = []
        num_copies_to_go = lambda_ - len(offspring)
        # makes 2*num_to_go, with a min of 8 individuals (threading).
        for i in range(max(4, num_copies_to_go)):
            op_choice = random.random()
            # can't do crossover with fewer than one individual.
            if len(population) > 1 and op_choice < cxpb:  # Apply crossover
                ind1, ind2 = list(map(toolbox.clone, random.sample(population, 2)))
                ind1, ind2 = toolbox.mate(ind1, ind2)
                # just in case
                del ind1.parsimony
                del ind2.parsimony
                ind1.closest_features = None
                ind2.closest_features = None
                ind1.str = None
                ind2.str = None
                ind1.output = None
                ind2.output = None
                candidates.append(ind1)
                candidates.append(ind2)
            else:
                ind1 = toolbox.clone(random.choice(population))
                ind1, = toolbox.mutate(ind1)
                del ind1.parsimony
                ind1.closest_features = None
                ind1.str = None
                ind1.output = None
                candidates.append(ind1)
                ind2 = toolbox.clone(random.choice(population))
                ind2, = toolbox.mutate(ind2)
                del ind2.parsimony
                ind2.closest_features = None
                ind2.str = None
                ind2.output = None
                candidates.append(ind2)
        # it shouldn't select more than we need.
        num_produced = check_uniqueness_str(candidates, lambda_, offspring)

        # safeguard infinite loops
        if num_produced == 0:
            no_change += 1
        else:
            no_change = 0

    if len(offspring) < lambda_:
        num_copies = lambda_ - len(offspring)
        print('Only {} offspring produced, reproducing {} individuals to get to {}.'.format(len(offspring), num_copies,
                                                                                            lambda_))
        offspring.extend(list(map(toolbox.clone, random.sample(population, num_copies))))
    assert len(offspring) == lambda_, ('Must produce exactly {} offspring, not {}.'.format(lambda_, len(offspring)))
    return offspring


def eval_population(population, toolbox):
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    return invalid_ind


def eval_population_cached_simple(population, toolbox, eval_func, rundata, cache=0):
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    non_cached = []

    arrays = get_arrays_cached(invalid_ind, toolbox, rundata)

    # try and get from cache, and then only thread the leftovers
    for ind, dat_array in zip(invalid_ind, arrays):
        hashable = ArrayWrapper(dat_array)
        res = try_cache(rundata, hashable, cache)
        if res:
            ind.fitness.values = res[0]
            ind.closest_features = res[1]
        if not res:
            non_cached.append((ind, dat_array))

    arg1 = [x[1] for x in non_cached]
    arg2 = [x[0].str for x in non_cached]
    # we don't want to pass the individual between threads!
    outputs = toolbox.starmap(eval_func, zip(arg1, arg2))
    for non_cache, output in zip(non_cached, outputs):
        fitness = output[0],
        closest_features = output[1]
        non_cache[0].fitness.values = fitness
        non_cache[0].closest_features = closest_features
        # add to cache
        if cache >= 0:
            rundata.fitnessCache[cache][ArrayWrapper(non_cache[1])] = fitness, closest_features
            rundata.stores = rundata.stores + 1

    return non_cached
