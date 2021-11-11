# There should be no main() in this file!!! 
# Nothing should start running when you import this file somewhere.
# You may add other supporting functions to this file.
#
# Important rules:
# 1) Function pa_bc must return tensor which has dimensions (#a x #b x #c),
#    where #v is a number of different values of the variable v.
#    For input variables #v = how many input values of this variable you give to the function.
#    For output variables #v = number of all possible values of this variable.
#    Ex. for pb_a: #b = bmax-bmin+1,   #a is arbitrary.
# 2) Random variables in function names must be written in alphabetic order
#    e.g. pda_cb is an improper function name (pad_bc must be used instead)
# 3) Single dimension must be explicitly stated:
#    if you give only one value of a variable a to the function pb_a, i.e. #a=1, 
#    then the function pb_a must return tensor of shape (#b, 1), not (#b,).
#
# The format of all the functions for distributions is the following:
# Inputs:
# params - dictionary with keys 'amin', 'amax', 'bmin', 'bmax', 'p1', 'p2', 'p3'
# model - model number, number from 1 to 4
# all other parameters - values of the conditions (variables a, b, c, d).
#                        Numpy vectors of size (k,), where k is an arbitrary number.
#                        For variant 3: c and d must be numpy arrays of size (k,N),
#                        where N is a number of lectures.
# Outputs:
# prob, val
# prob - probabilities for different values of the output variable with different input conditions
#        prob[i,...] = p(v=val[i]|...)
# val - support of a distribution, numpy vector of size (#v,) for variable v
#
# Example 1:
#    Function pc_ab - distribution p(c|a,b)
#    Input: a of size (k_a,) and b of size (k_b,)
#    Result: prob of size (cmax-cmin+1,k_a,k_b), val of size (cmax-cmin+1,) 
#
# Example 2 (for variant 3):
#    Function pb_ad - distribution p(b|a,d_1,...,d_N)
#    Input: a of size (k_a,) and d of size (k_d,N)
#    Result: prob of size (bmax-bmin+1,k_a,k_d), val of size (bmax-bmin+1,)
#
# The format the generation function from variant 3 is the following:
# Inputs:
# N - how many points to generate
# all other inputs have the same format as earlier
# Outputs:
# d - generated values of d, numpy array of size (N,#a,#b)

import numpy as np
from scipy.stats import binom, poisson
from typing import List, Union


# supportive functions
def prior_a_b(_max: int, _min: int):
    rng = np.arange(_min, _max + 1)
    denominator = rng.shape[0]
    return rng, np.repeat(1 / denominator, denominator)


def c_max(params: dict):
    return params["amax"] + params["bmax"]


def p_c_ab(a: Union[List, np.ndarray], b: Union[List, np.ndarray], params: dict, model: int):

    assert model in [1, 2], "Undefined model!"

    a, b = a.reshape(-1, 1), b.reshape(1, -1)
    _c_max = c_max(params)
    k = np.arange(_c_max + 1)

    # binomial distribution
    if model == 1:
        # probability mass function (x, n, p)
        a_pmf = binom.pmf(k.reshape(1, -1), a, params["p1"])
        b_pmf = binom.pmf(k.reshape(-1, 1), b, params["p2"])

    # poisson distribution
    elif model == 2:
        a_pmf = poisson.pmf(k.reshape(1, -1), a * params["p1"])
        b_pmf = poisson.pmf(k.reshape(-1, 1), b * params["p2"])

    pc_ab = np.zeros((_c_max + 1, a.shape[0], b.shape[1]), dtype=np.float128)

    for i in range(_c_max + 1):
        pc_ab[i] = np.dot(a_pmf[:, :i + 1], b_pmf[:i + 1, :][::-1])

    return pc_ab, np.arange(_c_max + 1)


def pd_c(params: dict):
    c = np.arange(c_max(params) + 1).reshape(1, -1)
    d = np.arange(c_max(params) * 2 + 1).reshape(-1, 1)

    c_pmf = binom.pmf(d - c, c.reshape(1, -1).squeeze(), params["p3"])

    return c_pmf, d.reshape(1, -1).squeeze()


def pd_b(params: dict, model: int):
    b = pb(params, model)[0]
    _pd_c, d = pd_c(params)
    _pc_b = pc_b(b, params, model)[0]
    _pd_b = _pd_c @ _pc_b

    return _pd_b, d


def expectation(data_sample, distribution):
    return distribution.T @ data_sample


def variance(data_sample, distribution):
    return expectation(distribution, np.square(data_sample)) - np.square(expectation(distribution, data_sample))


# In variant 2 the following functions are required:
def pa(params: dict, model: int):
    return prior_a_b(params["amax"], params["amin"])


def pb(params: dict, model: int):
    return prior_a_b(params["bmax"], params["bmin"])


def pc(params: dict, model: int):
    a, _pa = pa(params, model)
    b, _pb = pb(params, model)

    pc_ab, c = p_c_ab(a, b, params, model)

    # pc_ab shape: [cmax + 1, _pa.shape[1], _pb.shape[1]]
    _pc = np.einsum("kij, i, j -> k", pc_ab, _pa, _pb)
    return _pc, c


def pd(params: dict, model: int):
    _pd_c, d = pd_c(params)
    _pc = pc(params, model)[0].reshape(1, -1)
    _pd = np.sum(_pd_c * _pc, axis=1)
    return _pd, d


def pc_a(a: Union[List, np.ndarray], params: dict, model: int):
    b, p_b = pb(params, model)
    pc_ab, c = p_c_ab(a, b, params, model)
    _pc_a = np.sum(pc_ab, axis=2) * p_b[0]
    return _pc_a, c


def pc_b(b: Union[List, np.ndarray], params: dict, model: int):
    a, p_a = pa(params, model)
    pc_ab, c = p_c_ab(a, b, params, model)
    _pc_b = np.sum(pc_ab, axis=1) * p_a[0]
    return _pc_b, c


def pb_a(a: Union[List, np.ndarray], params: dict, model: int):
    b, _pb = pb(params, model)
    _pb_a = np.full(shape=(b.shape[0], a.shape[0]), fill_value=_pb[0])
    return _pb_a, b


def pb_d(d: Union[List, np.ndarray], params: dict, model: int):
    b, _pb = pb(params, model)
    _pd_b = pd_b(params, model)[0][d]
    _pb_d = _pd_b.T * _pb[0]
    _pb_d /= np.sum(_pb_d, axis=0)

    return _pb_d, b


def pb_ad(a: Union[List, np.ndarray], d: List, params: dict, model: int):
    b, _pb = pb(params, model)
    _pd_c = pd_c(params)[0][d]
    _pc_ab = p_c_ab(a, b, params, model)[0]
    _pc_a = pc_a(a, params, model)[0]
    numerator = _pd_c.dot(_pc_ab.swapaxes(0, 1)) * _pb[0]
    denominator = _pd_c.dot(_pc_a)
    _pb_ad = np.transpose(numerator / denominator[:, :, np.newaxis], (2, 1, 0))

    return _pb_ad, b
