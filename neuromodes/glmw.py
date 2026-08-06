import numpy as np
import scipy.stats as sps
from types import SimpleNamespace

from neuromodes.eigen import EigenData
from neuromodes.stats import ssqw, solvew

def _glmw(
        X, # (n_vertices, n_predictors)
        Y, # (n_vertices, n_data)
        contrast, # (n_contrasts, n_predictors)
        mass, # (n_vertices, n_vertices)
):
    # 0. Input validation
    ved = EigenData(data=(X, Y), mass=mass)
    X, Y = ved.data
    mass = ved.mass
    C = np.atleast_2d(contrast)

    # 1. Fit GLM
    beta = solvew(X, Y, mass) # Shape: (n_predictors, n_data)
    
    # 2. Shared Data Metrics
    df = X.shape[0] - np.linalg.matrix_rank(X)

    ssr = ssqw(Y - X.dot(beta), mass)
    sigma_sq = ssr / df # finish ssr
    
    # 3. Contrast Analysis
    D = C.dot(beta)  # Shape: (n_contrasts, n_data)

    A = (mass.dot(X)).T.dot(X)
    G = C.dot(np.linalg.solve(A, C.T))  # Shape: (n_contrasts, n_contrasts)
    V = np.linalg.solve(G, D)

    Q = np.maximum(np.sum(D * V, axis=0), 0) / sigma_sq
    
    return Q, D, df  

# TODO consider making mass optional and warning (as in stats.py)
# TODO consider making that part of EigenData (flag to return speye mass if None input)
def ttestw(X, Y, contrast, mass, alternative='two-sided'):
    """Weighted t-test for one or two samples."""

    if contrast.shape[0] != 1:
        raise ValueError(
            f"ttestw requires exactly 1 contrast (1 degree of freedom). "
            f"Received {contrast.shape[0]}. Use ftestw for multi-row contrasts."
        )

    Q, D, df = _glmw(X, Y, contrast, mass)
    
    # Recover directional t-statistic: t = sign(D) * sqrt(Q)
    statistic = np.sign(D.flatten()) * np.sqrt(Q)

    if alternative == 'two-sided':
        pvalue = 2 * sps.t.sf(np.abs(statistic), df=df)
    elif alternative == 'greater':
        pvalue = sps.t.sf(statistic, df=df)
    elif alternative == 'less':
        pvalue = sps.t.cdf(statistic, df=df)
    else:
        raise ValueError(f"Invalid alternative: {alternative}. Expected 'two-sided', 'greater', or 'less'.")
       
    # return a SimpleNamespace for compatibility with sps.ttest_*
    # (sps returns a TTestResult object with statistic, pvalue, and df attributes)
    # we don't return a confidence_interval method (?yet)
    return SimpleNamespace(
        statistic=statistic, 
        pvalue=pvalue,
        df=df
    )

def ftestw(X, Y, contrast, mass):
    """Weighted F-test for one-way ANOVA and ANCOVA."""
    Q, _, df = _glmw(X, Y, contrast, mass)
    
    # Compute omnibus F-statistic: F = Q / rank(C)
    rank_C = np.linalg.matrix_rank(contrast)
    statistic = Q / rank_C
    pvalue = sps.f.sf(statistic, dfn=rank_C, dfd=df)
    
    # same return as f_oneway
    return statistic, pvalue
