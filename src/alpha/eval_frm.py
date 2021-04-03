import re

import numpy as np
from numba import njit, numba
from numba.typed import List


@njit
def _within_tol(a, b, rtol=1e-05, atol=1e-08):
    return np.less_equal(np.abs(a - b), atol + rtol * np.abs(b))


@njit
def np_isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    # from https://github.com/numba/numba/pull/4610/
    # Based on NumPy impl.
    # https://github.com/numpy/numpy/blob/d9b1e32cb8ef90d6b4a47853241db2a28146a57d/numpy/core/numeric.py#L2180-L2292

    xfin = np.asarray(np.isfinite(a))
    yfin = np.asarray(np.isfinite(b))
    if np.all(xfin) and np.all(yfin):
        return _within_tol(a, b, rtol, atol)
    else:
        r = _within_tol(a, b, rtol, atol)
        if equal_nan:
            return r | (~xfin & ~yfin)
        else:
            return r


feature_p = re.compile('f\\d+')


def sse_cost(embedding, ind, rd):
    num_cf = embedding.shape[1]
    feature_matches = [feature_p.findall(ind.str[d]) for d in range(num_cf)]
    sse = 0

    ind.closest_features = [None] * num_cf
    for d in range(num_cf):
        abs_diff = np.abs(rd.data_t - embedding[:, d])
        diffs = np.power(abs_diff, 2).sum(axis=1)
        ind.closest_features[d] = diffs.argmin()
        if 'f{}'.format(ind.closest_features[d]) in feature_matches[d]:
            return np.inf,
        else:
            sse += diffs[ind.closest_features[d]]

    return sse,


@njit
def _pearsons_cost(embedding, data_t_, int_features_in_tree, matching_features_array, matching_features_array_len):
    num_cf = embedding.shape[1]
    num_of = data_t_.shape[0]
    sum_abs_pear = 0.
    closest_features = np.empty((num_cf, num_of), dtype=numba.types.intp)
    #closest_features = np.empty((num_cf, num_of), dtype=int)

    for d in range(num_cf):
        if (embedding[:, d] == embedding[:, d][0]).all():
            # all constants...
            # worst case scenario I suppose? or inf?
            return np.inf, None
            # sum_abs_pear -= 1
        else:
            abs_pearson = np.zeros((num_of))
            for of in range(num_of):
                # for constant features, clearly zero correlation to them, no?
                if (data_t_[of, :] == data_t_[of, :][0]).all():
                    abs_pearson[of] = 0.
                else:
                    pearson = np.corrcoef(data_t_[of, :], embedding[:, d])[0, 1]
                    abs_pearson[of] = np.abs(pearson)

            closest_features[d] = abs_pearson.argsort()[::-1]
            closest_feat = closest_features[d][0]
            has_match = False
            matching_features_set = set(matching_features_array[closest_feat][:matching_features_array_len[closest_feat]])
            for feat in int_features_in_tree[d]:
                if feat in matching_features_set:
                    has_match = True
                    break
            if has_match:
                sum_abs_pear -= abs_pearson[closest_feat]
            else:
                sum_abs_pear += abs_pearson[closest_feat]

    fitness = (num_cf - sum_abs_pear)/num_cf
    return fitness, closest_features



def st_pearsons(data_t, matching_features_array,matching_features_array_len, embedding, strs):
    features_in_tree = [feature_p.findall(strs[d]) for d in range(embedding.shape[1])]
    int_features_in_tree = List(List(int(y.replace('f', '')) for y in x) for x in features_in_tree)

    output = _pearsons_cost(embedding, data_t, int_features_in_tree,matching_features_array,matching_features_array_len)

    return output
