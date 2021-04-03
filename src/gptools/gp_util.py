import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import glob, os

import argparse
import gzip as gz
import math
import random
import sys
from functools import partial
from pathlib import Path

import arff
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from deap import gp
from matplotlib.colors import LinearSegmentedColormap, to_hex
from scipy.special._ufuncs import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from gptools.multitree import str_ind
from gptools.read_data import read_data


def protectedDiv(left, right):
    if right == 0:
        return 1
    # try:
    return left / right
    # except ZeroDivisionError:
    #   return 1


def np_protectedDiv(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


def np_sigmoid(gamma):
    return expit(gamma)


def np_many_add(a, b, c, d, e):
    return a + b + c + d + e


def np_relu(x):
    return x * (x > 0)


def relu(x):
    # fast? https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
    return x * (x > 0)


def _if(a, b, c):
    return b if a < 0 else c


def np_if(a, b, c):
    return np.where(a < 0, b, c)


# np...??


def erc_array():
    return random.uniform(-1, 1)


# def local_search(ind,expr)
# return string

string_cache = set()


def add_to_string_cache(ind):
    hash = str_ind(ind)
    string_cache.add(hash)
    ind.str = hash


def check_uniqueness_str(inds, num_to_produce, offspring):
    num_uniques = 0
    for i in range(len(inds)):
        ind = inds[i]
        if len(offspring) == num_to_produce:
            break
        else:
            hash = str_ind(ind)
            if hash not in string_cache:
                string_cache.add(hash)
                ind.str = hash
                offspring.append(ind)
                del ind.fitness.values
                num_uniques += 1
    return num_uniques


clfs = {'KNN': KNeighborsClassifier(n_neighbors=3),
        'RF': RandomForestClassifier(random_state=0, n_estimators=100),
        }


def output_inds_simple(inds, toolbox, rd, suffix="", columns=None):
    output = np.hstack([evaluate_trees(rd.data_t, toolbox, ind) for ind in inds])
    if columns is None:
        columns = ['C' + str(i) for i in range(output.shape[1])]
    df = pd.DataFrame(output, columns=columns)
    df["class"] = rd.labels
    mean_fitness = np.mean([i.fitness.values[0] for i in inds])
    f_name = '{}-{:f}{}'.format(rd.dataset, mean_fitness, suffix)

    outfile = f_name + '.csv'
    p = Path(rd.outdir, outfile)
    df.to_csv(p, index=False)

    arff_file = f_name + '.arff'
    p = Path(rd.outdir, arff_file)

    str_class = [str(c) for c in rd.labels]
    unique_str_class = list(set(str_class))

    with p.open('w') as file:
        arff.dump({
            'description': u'',
            'relation': f_name,
            'attributes': [(c, 'REAL') for c in columns] + [('class', unique_str_class)],
            'data': np.hstack((output, np.asarray(str_class).reshape(-1, 1)))
        }, file)


def output_ind(ind, toolbox, rd, suffix="", compress=False, csv_file=None, tree_file=None, del_old=False,
               aug_file=False, columns=None, fitnesses=None):
    """ Does some stuff

    :param columns: optional list of column names for created features.
    :param aug_file: Whether to also save the augmented dataset
    :param ind: the GP Individual. Assumed two-objective
    :param toolbox: To evaluate the tree
    :param rd: dict-like object containing data_t (feature-major array), outdir (string-like),
    dataset (name, string-like), labels (1-n array of class labels)
    :param suffix: to go after the ".csv/tree"
    :param compress: boolean, compress outputs or not
    :param csv_file: optional path/buf to output csv to
    :param tree_file: optional path/buf to output tree to
    :param del_old: delete previous generations or not
    """
    old_files = glob.glob(rd.outdir + "*.tree" + ('.gz' if compress else ''))
    old_files += glob.glob(rd.outdir + "*.csv" + ('.gz' if compress else ''))
    output = evaluate_trees(rd.data_t, toolbox, ind)
    if columns is None:
        columns = ['C' + str(i) for i in range(output.shape[1])]
    df = pd.DataFrame(output, columns=columns)
    df["class"] = rd.labels

    compression = "gzip" if compress else None

    if fitnesses is None:
        fitnesses = ind.fitness.values

    def format_fitness(val):
        if isinstance(val, int):
            return '-' + str(val)
        elif isinstance(val, float):
            return '-{:f}'.format(val)
        else:
            return '-' + str(val)

    f_name = rd.dataset + ''.join([format_fitness(x) for x in fitnesses]) + suffix

    if csv_file:
        df.to_csv(csv_file, index=False)
    else:
        outfile = f_name + '.csv'
        if compress:
            outfile = outfile + '.gz'
        p = Path(rd.outdir, outfile)
        df.to_csv(p, index=False, compression=compression)

    if aug_file:
        outfile = f_name + '-aug.csv'
        combined_array = np.concatenate((output, rd.data), axis=1)
        aug_columns = columns + ['X' + str(i) for i in range(rd.data.shape[1])]
        df_aug = pd.DataFrame(combined_array, columns=aug_columns)
        df_aug["class"] = rd.labels
        if compress:
            outfile = outfile + '.gz'
        p = Path(rd.outdir, outfile)
        df_aug.to_csv(p, index=False, compression=compression)

    if tree_file:
        tree_file.write(str(ind[0]))
        for i in range(1, len(ind)):
            tree_file.write('\n')
            tree_file.write(str(ind[i]))
    else:
        outfile = f_name + '.tree'
        if compress:
            outfile = outfile + '.gz'

        p = Path(rd.outdir, outfile)
        with gz.open(p, 'wt') if compress else open(p, 'wt') as file:
            file.write(str(ind[0]))
            for i in range(1, len(ind)):
                file.write('\n')
                file.write(str(ind[i]))

    if del_old:
        # print(old_files)
        for f in old_files:
            try:
                os.remove(f)
            except OSError as e:  # if failed, report it back to the user ##
                print("Error: %s - %s." % (e.filename, e.strerror))


def explore_tree_recursive(subtree_root, indent, tree, toolbox, rd, labels, fitnesses, cost_function,
                           min_fitness=0, max_fitness=1):
    num_instances = rd.data_t.shape[1]
    sliced = tree.searchSubtree(subtree_root)
    output = toolbox.compile(gp.PrimitiveTree(tree[sliced]))(*rd.data_t)
    if isinstance(output, float):
        output = np.repeat(output, num_instances)
    # else:
    output = output.reshape((num_instances, -1))
    # diff = np.mean(np.abs(comparison_output - output))

    diff = cost_function(output, [str(tree)])[0]
    fitnesses[subtree_root] = (max_fitness - diff) / (max_fitness - min_fitness)
    labels[subtree_root] = labels[subtree_root] + '\n({:.2f})'.format(diff)
    # print('{}{} ({:.2f})'.format(indent, tree[subtree_root].name, diff))

    # print('{}{} - {}'.format(indent, start, stop))
    this_arity = tree[subtree_root].arity
    children = []
    i = 0
    idx = subtree_root + 1
    while i < this_arity:
        child_slice = tree.searchSubtree(idx)
        children.append([child_slice.start, child_slice.stop])
        i += 1
        idx = child_slice.stop

    # print('{}Children: {}'.format(indent, children))
    for child in children:
        explore_tree_recursive(child[0], indent + '\t', tree, toolbox, rd, labels, fitnesses, cost_function)


pretty_names = {
    'vdiv': '÷',
    'vmul': '×',
    'vadd': '+',
    'vsub': '–',
    'np_if': 'if'
}


def draw_ind(ind, toolbox, rd, cost_function, suffix=""):
    f_name = ('{}' + ('-{:f}' * len(ind.fitness.values)) + '{}').format(rd.dataset, *ind.fitness.values, suffix)
    # should be only one tree, no?
    assert len(ind) == 1
    tree = ind[0]
    outfile = f_name + '-tree{}.png'.format(0)
    nodes, edges, labels = gp.graph(tree)
    fitnesses = {}
    explore_tree_recursive(0, '', tree, toolbox, rd, labels, fitnesses, cost_function, max_fitness=2)

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.drawing.nx_pydot.graphviz_layout(g, prog="dot")

    cmap = LinearSegmentedColormap.from_list('', ['red', 'orange', 'green'])

    colors = [to_hex(cmap(fitnesses[i])) for i in nodes]
    for i in range(len(labels)):
        for l in pretty_names:
            labels[i] = labels[i].replace(l, pretty_names[l])

    fig, ax = plt.subplots()
    nx.draw_networkx(g, pos=pos, ax=ax, node_color=colors, labels=labels, node_size=1000)
    p = Path(rd.outdir, outfile)

    ##Try nxpd
    ax.autoscale(enable=True)
    fig.tight_layout()
    ax.axis("off")
    fig.savefig(p, bbox_inches='tight', dpi=100)
    plt.close(fig=fig)
    # g.node_attr['style'] = 'filled'


def evaluate_trees_with_compiler(data_t, compiler, individual):
    num_instances = data_t.shape[1]
    num_trees = len(individual)

    if not individual.str or len(individual.str) == 0:
        raise ValueError(individual)

    # result = []
    result = np.zeros(shape=(num_trees, num_instances))

    for i, f in enumerate(individual.str):
        # Transform the tree expression in a callable function
        func = compiler(expr=f)
        comp = func(*data_t)
        if (not isinstance(comp, np.ndarray)) or comp.ndim == 0:
            # it decided to just give us a constant back...
            comp = np.repeat(comp, num_instances)

        result[i] = comp
        # result.append(comp)

    # dat_array = np.array(result).transpose()
    dat_array = result.T
    return dat_array


def evaluate_trees(data_t, toolbox, individual):
    return evaluate_trees_with_compiler(data_t, toolbox.compile, individual)


def eval_if_not_cached(toolbox, data_t, ind):
    if (not hasattr(ind, 'output')) or (ind.output is None):
        ind.output = eval_tree_wrapper(toolbox, data_t, ind)
    return ind.output


def eval_tree_wrapper(toolbox, data_t, ind):
    return evaluate_trees(data_t, toolbox, ind)


def update_experiment_data(data, ns):
    dict = vars(ns)
    for i in dict:
        if dict[i] is not None:
            setattr(data, i, dict[i])
        else:
            if not hasattr(data, i):
                setattr(data, i, None)
        # data[i] = dict[i]


warnOnce = False


def try_cache(rundata, hashable, index=0, loggable=True):
    if index == -1:
        return
    if loggable:
        rundata.accesses = rundata.accesses + 1

    res = rundata.fitnessCache[index].get(hashable)
    if rundata.accesses % 1000 == 0:
        print("Caches size: " + str(rundata.stores) + ", Accesses: " + str(
            rundata.accesses) + " ({:.2f}% hit rate)".format(
            (rundata.accesses - rundata.stores) * 100 / rundata.accesses))
    return res

def arg_parse_helper(*args, **kwargs):
    """ Helps to build additional command line arguments as a list to pass into init_data.
    kwargs should contain 'arg_list' if a list has already been made (e.g. if this isn't the first time that
    this method is being called)
    """

    if 'arg_list' in kwargs:
        # don't add the arg_list..
        copied = kwargs.copy()
        copied.pop('arg_list')
        kwargs['arg_list'].append([args, copied])
        # just so you can chain things
        return kwargs['arg_list']
    else:
        return [[args, kwargs]]


def init_data(rd, additional_arguments=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--logfile", help="log file path", type=str, default="log.out")
    parser.add_argument("-d", "--dataset", help="Dataset file name", type=str, default="wine")
    parser.add_argument("--dir", help="Dataset directory", type=str,
                        default="/home/lensenandr/datasetsPy/")
    parser.add_argument("-g", "--gens", help="Number of generations", type=int)
    parser.add_argument("-od", "--outdir", help="Output directory", type=str, default="./")
    parser.add_argument("--erc", dest='use_ercs', help="Use Ephemeral Random Constants?", action='store_true')
    parser.add_argument("--trees", dest="max_trees", help="How many (or maximum if dynamic) trees to use", type=int)
    parser.add_argument("-cr", dest="cxpb", help="crossover rate", type=float)
    parser.add_argument("-mr", dest="mutpb", help="mutation rate", type=float)
    parser.add_argument("-p", dest="pop_size", help="population size", type=int, default=100)
    parser.add_argument("-e", dest="elitism", help="top-n elitism rae", type=int, default=10)
    parser.add_argument('-ef', "--excluded-functions", help="Functions to exclude from the function set (if any)",
                        nargs='+', dest='excluded_functions')
    parser.add_argument("-threads", help="Number of threads to use", dest="threads", type=int, default=1)
    parser.add_argument("-cae", "--classify-at-end", help="Enables performing k-fold classification at the end.",
                        dest="classify_at_end", action="store_true")
    if additional_arguments:
        for arg in additional_arguments:
            # this'll get set in rundata
            parser.add_argument(*arg[0], **arg[1])

    args = parser.parse_args()
    print(args)
    update_experiment_data(rd, args)

    DEL_PREV = False

    if (rd.outdir):
        if len(list(Path(rd.outdir).glob('*{}*1000.tree'.format(rd.dataset)))) > 0:
            if DEL_PREV:
                for p in Path(rd.outdir).glob('*{}*'.format(rd.dataset)):
                    if p.suffix in ['.csv', '.tree', '.arff']:
                        p.unlink()
            else:
                print('Skipping as outdir already has completed file')
                sys.exit(0)

    file = Path(args.dir) / (args.dataset + '.data')
    all_data = read_data(file)
    data = all_data["data"]
    rd.num_instances = data.shape[0]
    rd.num_features = data.shape[1]
    rd.labels = all_data["labels"]
    rd.data = data
    rd.data_t = data.T


def final_output(hof, toolbox, logbook, pop, rundata, classify=False, plot_curve=False):
    for res in hof:
        output_ind(res, toolbox, rundata, compress=False)
    p = Path(rundata.outdir, rundata.logfile + '.gz')
    with gz.open(p, 'wt') as file:
        file.write(str(logbook))
    pop_stats = [str(p.fitness) for p in pop]
    pop_stats.sort()
    hof_stats = [str(h.fitness) for h in hof]
    print("POP:")
    print("\n".join(pop_stats))
    print("PF:")
    print("\n".join(hof_stats))

    if plot_curve:
        gen = logbook.chapters['basic'].select("gen")
        gen = [float(x) for x in gen]
        fit_avgs = logbook.chapters["raw_cost"].select("median")
        size_avgs = logbook.chapters["size"].select("median")

        import matplotlib.pyplot as plt

        fig, host = plt.subplots()
        line1 = host.scatter(gen, fit_avgs, c='blue', label="Median Fitness")
        host.set_xlabel("Generation")
        host.set_ylabel("Fitness", color="b")
        for tl in host.get_yticklabels():
            tl.set_color("b")

        par1 = host.twinx()
        line2 = par1.scatter(gen, size_avgs, c='red', label="Median Size")
        par1.set_ylabel("Size", color="r")
        for tl in par1.get_yticklabels():
            tl.set_color("r")
        lns = [line1, line2]
        labs = [l.get_label() for l in lns]
        host.legend(lns, labs, loc="center right")
        p = Path(rundata.outdir, 'convergencePlot.pdf')

        plt.savefig(p)


def get_arrays_cached(invalid_ind, toolbox, rundata):
    for ind in invalid_ind:
        if (not hasattr(ind, 'str')) or (ind.str is None):
            add_to_string_cache(ind)

    proxy = partial(eval_if_not_cached, toolbox, rundata.data_t)
    arrays = list(map(proxy, invalid_ind))

    return arrays
