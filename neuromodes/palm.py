from warnings import warn
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.stats import genpareto

#from IPython import embed


def palm_pareto(G, Gdist, rev, Pthr, G1out):
    """
    Compute the p-values for a set of statistics G, taking
    as reference a set of observed values for G, from which
    the empirical cumulative distribution function (cdf) is
    generated. If the p-values are below Pthr, these are
    refined further using a tail approximation from the
    Generalised Pareto Distribution (GPD).

    Usage:
    P, apar, kpar, upar = palm_pareto(G, Gdist, rev, Pthr, G1out)

    Inputs:
    G      : Vector of Nx1 statistics to be converted to p-values
    Gdist  : A Mx1 vector of observed values for the same statistic
             from which the empirical cdf is build and p-values
             obtained. It doesn't have to be sorted.
    rev    : If true, indicates that the smallest values in G and
             Gvals, rather than the largest, are the most significant.
    Pthr   : P-values below this will be refined using GPD tail.
    G1out  : Boolean indicating whether G1 should be removed from the null
             distribution.

    Output:
    P      : P-values.
    apar   : Scale parameter of the GPD.
    kpar   : Shape parameter of the GPD.
    upar   : Location parameter of the GPD.
    """

    # Ensure inputs are numpy arrays

    # these should be vectors
    G = np.asarray(G)
    if G.ndim <= 1:
        G = G.reshape(-1)
        
    Gdist = np.asarray(Gdist)
    if Gdist.ndim <= 1:
        Gdist = Gdist.reshape(-1)

    if G1out:
        Gdist = Gdist[1:, :]
    
    # Compute the usual permutation p-values.
    P = palm_datapval(G, Gdist, rev)

    # if none of the p-values are below the threshold, return early
    Pidx = P < Pthr  # don't replace this "<" for "<=".
    if not np.any(Pidx):
        return P, np.nan, np.nan, np.nan
        
    # Number of permutations & distribution CDF
    nP = Gdist.shape[0] # TODO replace with Gdist.size ?
    if rev:
        _, Gdist_sorted, Gcdf = palm_competitive(Gdist, 'descend', True)
    else:
        _, Gdist_sorted, Gcdf = palm_competitive(Gdist, 'ascend', True)
    
    # Flatten for indexing
    Gdist_sorted = Gdist_sorted.flatten()
    Gcdf = Gcdf.flatten() / nP  
    
    # Keep adjusting until the fit is good. Change the step to 10 to get
    # the same result as Knijnenburg et al.
    factor = -1 if rev else 1
    for q, Q in enumerate(np.arange(751, 1000, 10) / 1000.0):
        
        # Find where the tail starts and thus upar
        qidx = Gcdf >= Q
        qi_idx = np.where(qidx)[0][0]
        if qi_idx == 0:
            upar = (Gdist_sorted[qi_idx] - Gdist_sorted[qi_idx + 1]) / 2
        else:
            upar = (Gdist_sorted[qi_idx] + Gdist_sorted[qi_idx - 1]) / 2


        # Estimate the distribution parameters. See Section 3.2 of Hosking &
        # Wallis (1987). Compared to the usual GPD parameterisation, 
        # here k = shape (xi), and a = scale.
        Gtail = Gdist_sorted[qidx] # finish qidx
        ytail = factor * (Gtail - upar)
        x = np.mean(ytail)
        s2 = np.var(ytail, ddof=1)

        # Check for zero/degenerate variance in the tail (e.g., discrete ties)
        if s2 <= 1e-12:
            warn(
                f"Zero or near-zero variance encountered in tail at quantile Q={Q:.3f}. "
                "Skipping to next threshold.",
                UserWarning
            )
            continue

        # Check if the fitness is good
        apar = x * (x**2 / s2 + 1) / 2
        kpar = (x**2 / s2 - 1) / 2 # finish x, s2
        # TODO fix the sort here
        A2pval = andersondarling(np.sort(gpdpvals(ytail, apar, kpar)), kpar)
            
        # If yes, return. If not, try again with the next quantile.
        if A2pval > 0.05:
            mask_y = (factor * (G - upar) > 0) & Pidx
            y = factor * (G[mask_y] - upar)
            P[mask_y] = gpdpvals(y, apar, kpar) * Gtail.size / nP
            return P, apar, kpar, upar
    
    # If above loop doesn't return, GPD tail approximation failed to converge
    warn(
        "GPD tail approximation failed to converge. "
        "Returning permutation p-values for all statistics.",
        UserWarning
    )
    return P, np.nan, np.nan, np.nan

def gpdpvals(x, a, k):
    return genpareto.sf(x, c=-k, scale=a)  

def andersondarling(z, k):
    """
    Compute the Anderson-Darling statistic and return an
    approximated p-value based on the tables provided in:
    * Choulakian V, Stephens M A. Goodness-of-Fit Tests
      for the Generalized Pareto Distribution. Technometrics.
      2001;43(4):478-484.
    
    Copies the MATLAB implementation. 
    TODO : check which case should be used -- appears that k is known?
    TODO : consider replacing k with -k -- as in the paper
    """
    # Prelims/constants
    # Table 2 of the paper (Case 3: a and k unknown, bold values)
    K_TABLE = np.array([0.9, 0.5, 0.2, 0.1, 0, -0.1, -0.2, -0.3, -0.4, -0.5])
    P_TABLE = np.array([0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001])
    A2_TABLE = np.array([
        [0.3390, 0.4710, 0.6410, 0.7710, 0.9050, 1.0860, 1.2260, 1.5590],
        [0.3560, 0.4990, 0.6850, 0.8300, 0.9780, 1.1800, 1.3360, 1.7070],
        [0.3760, 0.5340, 0.7410, 0.9030, 1.0690, 1.2960, 1.4710, 1.8930],
        [0.3860, 0.5500, 0.7660, 0.9350, 1.1100, 1.3480, 1.5320, 1.9660],
        [0.3970, 0.5690, 0.7960, 0.9740, 1.1580, 1.4090, 1.6030, 2.0640],
        [0.4100, 0.5910, 0.8310, 1.0200, 1.2150, 1.4810, 1.6870, 2.1760],
        [0.4260, 0.6170, 0.8730, 1.0740, 1.2830, 1.5670, 1.7880, 2.3140],
        [0.4450, 0.6490, 0.9240, 1.1400, 1.3650, 1.6720, 1.9090, 2.4750],
        [0.4680, 0.6880, 0.9850, 1.2210, 1.4650, 1.7990, 2.0580, 2.6740],
        [0.4960, 0.7350, 1.0610, 1.3210, 1.5900, 1.9580, 2.2430, 2.9220]
    ])

    # Input validation
    # TODO check sort order here. does it need to be ascending?
    z = np.asarray(z)
    if z.ndim != 1: 
        raise ValueError("Input array 'z' must be one-dimensional.")
    if not np.all(z[:-1] <= z[1:]):
        raise ValueError("Input array 'z' must be sorted in ascending order.")
    n = z.size

    k = float(k)
    if k < np.min(K_TABLE) or k > np.max(K_TABLE):
        warn(
            f"Shape parameter k={k} is outside the table bounds [{np.min(K_TABLE)}, {np.max(K_TABLE)}]. "
            "Critical values will be extrapolated and may be unreliable.",
            UserWarning
        )
    
    # Set up interpolation functions for the table
    # TODO add extrapolate keyword
    # TODO find best way to fix the sign of k (see above)
    # Interpolate critical values for k and then p-values for A2
    k_to_a2 = make_interp_spline(-K_TABLE, A2_TABLE, axis=0, k=1)
    a2_to_p = make_interp_spline(k_to_a2(-k), P_TABLE, k=1, axis=0)

    # Main computation
    A2 = -n - np.mean( np.log( z*(1-z[::-1]) ) * np.arange(1,2*n,2) )
    p = a2_to_p(A2)
    
    return np.clip(p, 0.0, 1.0)


def palm_datapval(G, Gvals, rev=False):
    """Compute p-values for statistics G given observed reference values Gvals."""
    
    # Input validation
    G = np.asarray(G)
    Gvals = np.asarray(Gvals)
    rev = bool(rev) # note that rev is more like ord than mod
    # TODO consider changing rev and ord to 'descending' like np.sort

    # Compute empirical distribution thresholds and probabilities
    if rev: 
        _, cdfG, distp = palm_competitive(Gvals.ravel(), ord='ascend', mod=True)
    else: 
        _, cdfG, distp = palm_competitive(Gvals.ravel(), ord='descend', mod=True)
    
    # _, cdfG, distp = palm_competitive(Gvals.ravel(), ord='ascend', mod=rev)
    # if not rev: 
    #     cdfG = np.flipud(cdfG)
    #     distp = np.flipud(distp.size - distp + 1)

    cdfG, u_idx = np.unique(cdfG, return_index=True)
    distp = distp[u_idx] / distp.size
        
    # Pad probability values: 0 at start if rev=True, 0 at end if rev=False
    values = np.concatenate(([1-rev], distp, [rev]))
    bins = np.digitize(G, cdfG, right=not rev) + (1-rev)

    return np.clip(values[bins], 0.0, 1.0)


def palm_competitive(X, ord='ascend', mod=False):
    """
    Sort a set of values and return their competition
    ranks (standard 1224 or modified 1334).
    Currently copies MATLAB implementation. 
    Only supports sorting/ranking along axis=0. 
    """

    # Input validation
    X = np.asarray(X)
    if np.any(np.isnan(X)):
        raise ValueError("Data cannot be sorted. Check for NaNs in input.")
    if np.any(np.all(np.isinf(X), axis=0)):
        raise ValueError("Data cannot be sorted. Maximum statistic is +Inf or -Inf for all permutations.")
    if ord not in ['ascend', 'descend']:
        raise ValueError("Invalid 'ord' parameter. Must be 'ascend' or 'descend'.")
    if not isinstance(mod, bool):
        raise ValueError("Invalid 'mod' parameter. Must be a boolean (True or False).")

    # 1. Sort data in ascending/descending order (if mod: need to reverse direction)
    Y = -X if ((ord == 'descend') ^ (mod)) else X
    sortidx = np.argsort(Y, axis=0) # Y finished

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
        S = np.flipud(S)
        srtR = np.flipud(nR - srtR + 1)

    return unsrtR, S, srtR
