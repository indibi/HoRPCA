
import numpy as np
from numpy.linalg import norm
from util.merge_Tucker import merge_Tucker
from util.t2m import t2m
from util.m2t import m2t

def T_rank(X, sv_threshold=10**-10):
    """ Calculate tucker rank. Given tensor must be of order at least greater than two.

    Args:
        X (np.ndarray): Given tensor to calculate its tucker rank.

        normalized (boolean): If it is set to True, function returns the normalized
        rank of the tensor, mentioned in,

        normalized_rank = ||1./X.shape||_{1/2}  ||trank(X)||_{1/2}
    Tomioka, Suzuki et. al. "Statistical performance of convex tensor decomposition."
    In Advances in Neural Information Processing Systems 25 (2011)


    Returns:
        tranks: List of ranks in each mode.
    """
    if type(X) != np.ma and type(X) != np.array and type(X) != np.ndarray:
        raise TypeError("T_rank: Parameter X is not a numpy tensor!")
    
    order = len(X.shape)
    if order <2:
        raise ValueError("T_rank: X cannot be a vector!")

    tranks=np.zeros(order)
    s=[]
    for _ in range(len(X.shape)):
        s.append( np.linalg.svd(t2m(X,_),full_matrices=False, compute_uv=False) )
        # Unfortunately svd function is as accurate as digital computers can be and returns calculates
        # singular values that should be zero close to 10**-16 but not zero. We threshold these values 
        s[-1] = s[-1][np.where( s[-1] > sv_threshold )]       
        tranks[_]=len(s[-1])

    nrank = X.shape
    sqrt_invn= [np.sqrt(1/x) for x in nrank] # elemenwise square root of the inverse of the dimensions
    sqrt_tranks = np.sqrt(tranks)            # elementwise square roots of the n mode ranks
    norm_rank= ((sqrt_tranks.sum()/len(sqrt_tranks))**2) * ((sum(sqrt_invn)/len(sqrt_invn))**2)
    return (tranks, norm_rank)
