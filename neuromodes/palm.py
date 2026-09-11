from warnings import warn
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.stats import genpareto

def palm_pareto(G, 
                Gdist, 
                descending=False, 
                Pthr=0.05, # TODO rename to Ptest or Pnaive or similar. consider adding tests
                G1out=False, 
                thresholds: float | np.ndarray = np.arange(751, 1000, 10) / 1000.0, # TODO rename to thresholds_to_test or similar
                Paccept=0.05 # TODO consider adding tests for this parameter
                ): 

    # 0. Input validation: ensure inputs are 1D numpy arrays (for now)
    G = np.asarray(G)
    if G.ndim > 1:
        raise ValueError("G must be a scalar or 1D array.")
    
    Gdist = np.asarray(Gdist)
    if Gdist.ndim != 1:
        raise ValueError("Gdist must be a 1D array.")
    Gdist = Gdist[G1out:] # remove G1 if requested # TODO remove this?

    # 1. Convert G to p-values using the empirical distribution of Gdist
    # If none of the p-values are below the threshold, return early
    P = palm_datapval(G, Gdist, descending=descending)
    Pidx = P < Pthr  # don't replace this "<" for "<=".
    if not np.any(Pidx):
        return P, np.nan, np.nan, np.nan

    # 2. Find the putative tail of the distribution and the corresponding GPD parameters
    _, Gdist_sorted, Gcdf = palm_competitive(Gdist, descending=descending, ranking='modified')
    Gcdf = Gcdf / Gcdf.size

    goodthresh, apar, kpar, upar = np.nan, np.nan, np.nan, np.nan
    for thr in np.atleast_1d(thresholds):
        A2pval, apar, kpar, upar = _fit_tail(Gdist_sorted, np.argmax(Gcdf >= thr))
        if A2pval > Paccept:
            goodthresh = thr
            break

    # 3. If a suitable tail was found, refine the p-values using the GPD approximation
    if not np.isnan(goodthresh):
        diffG = upar - G if descending else G - upar
        mask = Pidx & (diffG > 0)
        newP = _gpdpvals(diffG, apar, kpar) * np.mean(Gcdf >= goodthresh) # proportion in the tail
        P = np.where(mask, newP, P)
    else: 
        warn(
            "GPD tail approximation failed to converge. "
            "Returning permutation p-values for all statistics.",
            UserWarning
        )

    return P, apar, kpar, upar

def _fit_tail(Gdist_sorted, tailstartidx):
    if tailstartidx != 0: # if tail does not start at the first element, use the mean
        upar = (Gdist_sorted[tailstartidx] + Gdist_sorted[tailstartidx - 1]) / 2
    else: # Extrapolate a half-step "before" the first boundary point i.e. $a-(b-a)/2$
        upar = (3 * Gdist_sorted[0] - Gdist_sorted[1]) / 2 # fixes a bug in the original MATLAB

    # Estimate the distribution parameters. See Section 3.2 of Hosking &
    # Wallis (1987). Compared to the usual GPD parameterisation, 
    # here k = shape (xi), and a = scale.
   
    # use abs as we just need to flip if descending
    # equivalent to 
    # `ytail = upar - Gtail if descending else Gtail - upar`
    ytail = np.abs(Gdist_sorted[tailstartidx:] - upar) # always increasing

    # Check if the fitness is good
    x = np.mean(ytail)
    s2 = np.var(ytail, ddof=1)

    apar = x * (x**2 / s2 + 1) / 2
    kpar = -(x**2 / s2 - 1) / 2 # finish x, s2

    z = _gpdpvals(ytail, apar, kpar) # 1-cdf so this makes ytail decreasing
    A2pval, _ = andersondarling(np.flip(z), kpar) # flip back to increasing

    return A2pval, apar, kpar, upar

def _gpdpvals(y, a, k):
    return genpareto.sf(y, scale=a, c=k)

def palm_datapval(G, Gvals, descending=False): #TODO rename to Gdist
    """Compute p-values for statistics G given observed reference values Gvals."""
    # Input validation
    G = np.asarray(G)

    Gvals = np.asarray(Gvals) # Only 1D data for Gvals at the moment. Add `axis` in future?
    if Gvals.ndim != 1:
        raise ValueError("Gvals must be a 1D array of observed values.")
    N = Gvals.size

    # Main computation
    idx = np.argsort(Gvals) # must be ascending for seachsorted
    if descending:
        # Left-tailed: count elements in Gvals <= g
        # 'side=right' finds insertion point i where Gvals_sorted[:i] <= g
        pvals = np.searchsorted(Gvals, G, side='right', sorter=idx) / N
    else:
        # Right-tailed: count elements in Gvals >= g
        # 'side=left' finds insertion point i where Gvals_sorted[i:] >= g
        pvals = 1 - np.searchsorted(Gvals, G, side='left', sorter=idx) / N
    return np.clip(pvals, 0.0, 1.0)


def palm_competitive(X, descending=False, ranking='standard'):
    """
    Sort a set of values and return their competition
    ranks (standard 1224 or modified 1334).
    Only supports sorting/ranking along axis=0. 
    """

    # Input validation
    X = np.asarray(X)
    if np.any(np.isnan(X)):
        raise ValueError("Data cannot be sorted. Check for NaNs in input.")
    if np.any(np.all(np.isinf(X), axis=0)):
        raise ValueError("Data cannot be sorted. Maximum statistic is +Inf or -Inf for all permutations.")
    if not isinstance(descending, bool):
        raise ValueError("Invalid 'descending' parameter. Must be a boolean.")
    if ranking not in {'standard', 'modified'}:
        raise ValueError("Invalid 'ranking' parameter. Must be 'standard' or 'modified'.")

    # 1. Sort data in ascending/descending order (if mod: need to reverse direction)
    mod = (ranking == 'modified')
    factor = -1 if (descending ^ mod) else 1
    sortidx = np.argsort(factor * X, axis=0) # factor finished

    # 2. Detect ties and assign ranks
    # base_ranks is a vector of the form (1,2,3,...,nR) of size (nR,1,1,...,1) for broadcasting
    nR = X.shape[0]
    base_ranks = np.expand_dims(np.arange(nR) + 1, axis=tuple(range(1,X.ndim)))

    # Find ties along axis 0
    S = np.take_along_axis(X, sortidx, axis=0)
    is_tie = np.concatenate([
        np.zeros_like(S[:1], dtype=bool), # a row of all False to start
        S[1:] == S[:-1]
        ], axis=0)

    # Take the original rank OR the rank of the first occurrence of the tie, as needed
    srtR = np.maximum.accumulate(np.where(is_tie, 0, base_ranks), axis=0)   # base_ranks, is_tie finished

    # 3. Outputs: map sorted ranks back to original array order
    unsrtR = np.take_along_axis(srtR, np.argsort(sortidx, axis=0), axis=0)

    # Handle modified competitive ranking (1334 scheme)
    if mod:
        unsrtR = nR - unsrtR + 1
        S = np.flip(S, axis=0)
        srtR = np.flip(nR - srtR + 1, axis=0)

    return unsrtR, S, srtR


def andersondarling(z_ascending, k):
    """
    Compute the Anderson-Darling statistic and return an
    approximated p-value based on the tables provided in:
    * Choulakian V, Stephens M A. Goodness-of-Fit Tests
      for the Generalized Pareto Distribution. Technometrics.
      2001;43(4):478-484.
    """
    # Prelims/constants
    # Table 2 of the paper (Case 3: a and k unknown, bold values)
    K_TABLE = np.array([-0.90, -0.50, -0.20, -0.10,  0.00,  0.10,  0.20,  0.30,  0.40,  0.50])
    P_TABLE = np.array([0.500, 0.250, 0.100, 0.050, 0.025, 0.010, 0.005, 0.001])
    A2_TABLE = np.array([
        [0.339, 0.471, 0.641, 0.771, 0.905, 1.086, 1.226, 1.559],
        [0.356, 0.499, 0.685, 0.830, 0.978, 1.180, 1.336, 1.707],
        [0.376, 0.534, 0.741, 0.903, 1.069, 1.296, 1.471, 1.893],
        [0.386, 0.550, 0.766, 0.935, 1.110, 1.348, 1.532, 1.966],
        [0.397, 0.569, 0.796, 0.974, 1.158, 1.409, 1.603, 2.064],
        [0.410, 0.591, 0.831, 1.020, 1.215, 1.481, 1.687, 2.176],
        [0.426, 0.617, 0.873, 1.074, 1.283, 1.567, 1.788, 2.314],
        [0.445, 0.649, 0.924, 1.140, 1.365, 1.672, 1.909, 2.475],
        [0.468, 0.688, 0.985, 1.221, 1.465, 1.799, 2.058, 2.674],
        [0.496, 0.735, 1.061, 1.321, 1.590, 1.958, 2.243, 2.922]
    ])
    k = float(k)
    if k < np.min(K_TABLE) or k > np.max(K_TABLE):
        warn(
            f"Shape parameter k={k} is outside the table bounds [{np.min(K_TABLE)}, {np.max(K_TABLE)}]. "
            "Critical values will be extrapolated and may be unreliable.",
            UserWarning
        )

    # Main computation
    z = np.asarray(z_ascending)
    if z.ndim != 1: 
        raise ValueError("Input array 'z' must be one-dimensional.")
    if not np.all(np.diff(z) >= 0):
        raise ValueError("Input array 'z' must be sorted in ascending order.")
    n = z.size
    a2 = -n - np.mean( np.log( z*(1-z[::-1]) ) * np.arange(1,2*n,2) )
    
    # Interpolate critical values for k and then p-values for A2
    k_to_a2 = make_interp_spline(K_TABLE, A2_TABLE, axis=0, k=1)
    a2_to_p = make_interp_spline(k_to_a2(k), P_TABLE, k=1, axis=0)
    p = a2_to_p(a2)
    
    return np.clip(p, 0.0, 1.0), np.clip(a2, 0.0, np.inf)
