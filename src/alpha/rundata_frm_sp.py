import cachetools
from defaultlist import defaultlist

class RD(object):
    def __init__(self):
        self.data = None
        self.data_t = None
        self.labels = None
        self.outdir = None
        self.nobj = 1
        self.fitnessCache = defaultlist(lambda: cachetools.LRUCache(maxsize=1000000))
        self.accesses = 0
        self.stores = 0

        self.max_depth = 6  # 7#12#8
        self.pop_size = 100  # 1024#100
        self.cxpb = 0.8
        self.mutpb = 0.2
        self.max_trees = 1  # 5#2
        self.gens = 1000

        self.num_instances = 0
        self.num_features = 0

        self.max_species = 10#None

        self.alpha = 0.001

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


rd = RD()
