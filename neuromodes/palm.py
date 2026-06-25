import numpy as np
from scipy.interpolate import interp1d

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

    # Compute the usual permutation p-values.
    if G1out:
        Gdist = Gdist[1:, :]
    
    P = palm_datapval(G, Gdist, rev)
    Pidx = P < Pthr  # don't replace this "<" for "<=".

    apar = np.nan
    kpar = np.nan
    upar = np.nan

    # If some of these are small (as specified by the user), these
    # will be approximated via the GPD tail.
    if np.any(Pidx):
        
        # Number of permutations & distribution CDF
        nP = Gdist.shape[0]
        if rev:
            _, Gdist_sorted, Gcdf = palm_competitive(Gdist, 'descend', True)
        else:
            _, Gdist_sorted, Gcdf = palm_competitive(Gdist, 'ascend', True)
        
        # Flatten for indexing
        Gdist_sorted = Gdist_sorted.flatten()
        Gcdf = Gcdf.flatten() / nP
        
        # Keep adjusting until the fit is good. Change the step to 10 to get
        # the same result as Knijnenburg et al.
        Q = np.arange(751, 1000, 10) / 1000.0
        nQ = Q.size
        q = 0 # 0-indexed
        Ptail = np.nan
        
        while np.any(np.isnan(Ptail)) and q < nQ - 1 and np.unique(Gdist_sorted[Gcdf >= Q[q]]).size > 1:

            # Get the tail
            qidx = Gcdf >= Q[q]
            Gtail = Gdist_sorted[qidx]
         
            qi_idx = np.where(qidx)[0][0]
            if qi_idx == 0:
                upar = Gdist_sorted[qi_idx] - np.mean(Gdist_sorted[qi_idx : qi_idx + 2])
            else:
                upar = np.mean(Gdist_sorted[qi_idx - 1 : qi_idx + 1])
            
            if rev:
                mask_y = (G < upar) & Pidx
                ytail = upar - Gtail
                y = upar - G[mask_y]
            else:
                mask_y = (G > upar) & Pidx
                ytail = Gtail - upar
                y = G[mask_y] - upar
            
            # Estimate the distribution parameters. See Section 3.2 of Hosking &
            # Wallis (1987). Compared to the usual GPD parameterisation, 
            # here k = shape (xi), and a = scale.
            x = np.mean(ytail)
            s2 = np.var(ytail, ddof=1)
            apar = x * (x**2 / s2 + 1) / 2
            kpar = (x**2 / s2 - 1) / 2
            
            # Check if the fitness is good
            A2pval = andersondarling(gpdpvals(ytail, apar, kpar), kpar)
                
            # If yes, keep. If not, try again with the next quantile.
            if A2pval > 0.05:
                cte = Gtail.size / nP
                Ptail = cte * gpdpvals(y, apar, kpar)
            else:
                q = q + 1
        
        # Replace the permutation p-value for the approximated p-value
        if not np.any(np.isnan(Ptail)):
            if rev:
                P[(G < upar) & Pidx] = Ptail
            else:
                P[(G > upar) & Pidx] = Ptail
    return P, apar, kpar, upar

def gpdpvals(x, a, k):
    """
    Compute the p-values for a GPD with parameters a (scale)
    and k (shape).
    """
    x = np.asarray(x)
    eps = np.finfo(float).eps
    if np.abs(k) < eps:
        p = np.exp(-x / a)
    else:
        # Use complex power or handle potential negative base if necessary, 
        # though x should be restricted by the distribution support.
        p = np.maximum(0, (1 - k * x / a))**(1 / k)
    
    if k > 0:
        p[x > a / k] = 0
        
    return p

def andersondarling(z, k):
    """
    Compute the Anderson-Darling statistic and return an
    approximated p-value based on the tables provided in:
    * Choulakian V, Stephens M A. Goodness-of-Fit Tests
      for the Generalized Pareto Distribution. Technometrics.
      2001;43(4):478-484.
    """
    # Table 2 of the paper (Case 3: a and k unknown, bold values)
    ktable = np.array([0.9, 0.5, 0.2, 0.1, 0, -0.1, -0.2, -0.3, -0.4, -0.5])
    ptable = np.array([0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001])
    A2table = np.array([
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

    k = max(-0.5, min(0.9, k)) # Limit k to table range for robustness
    z = np.sort(np.asarray(z).flatten()) # AD statistic expects sorted z
    n = z.size
    j = np.arange(1, n + 1)

    # Anderson-Darling statistic:
    # A2 = -n -(1/n)*((2*j-1)*(log(z) + log(1-z(n+1-j)))')
    # Use small epsilon to avoid log(0)
    eps = np.finfo(float).eps
    z = np.clip(z, eps, 1 - eps)
    A2 = -n - (1.0 / n) * np.sum((2 * j - 1) * (np.log(z) + np.log(1 - z[n - j])))

    # Interpolate critical values for k (ktable is descending, so we flip)
    f_i1 = interp1d(ktable[::-1], A2table[::-1], axis=0, kind='linear', fill_value='extrapolate')
    i1 = f_i1(k)
    
    # Interpolate p-value for A2 (i1 are critical values, which are ascending)
    f_i2 = interp1d(i1, ptable, kind='linear', fill_value='extrapolate')
    A2pval = f_i2(A2)
    
    return np.clip(float(A2pval), 0, 1)

def palm_datapval(G, Gvals, rev):
    """
    Compute the p-values for a set of statistics G, taking
    as reference a set of observed values for G.
    """
    G = np.asarray(G)
    Gvals = np.asarray(Gvals).flatten()
    
    if rev: # if small G are significant
        # Sort the data and compute the empirical distribution
        _, cdfG_raw, distp_raw = palm_competitive(Gvals.reshape(-1, 1), 'ascend', True)
        cdfG, idx = np.unique(cdfG_raw, return_index=True)
        distp = distp_raw.flatten()[idx] / Gvals.size
        
        # Convert the data to p-values
        
        if G.size == 1:
            I = np.where(G >= cdfG)[0]
            if I.size > 0:
                pvals = np.array([distp[I[-1]]])
            else:
                
                pvals = np.array([1])
        else:
            pvals = np.ones(G.size)
            for z in range(G.size):
                I = np.where(G[z] >= cdfG)[0]
                if I.size > 0:
                    pvals[z] = distp[I[-1]]
            pvals = np.reshape(G.shape)
        # for g in range(cdfG.size):
        #     pvals[G >= cdfG[g]] = distp[g]
        # pvals[G > cdfG[-1]] = 1.0
            

    else: # if large G are significant (typical case)
        # Sort the data and compute the empirical distribution
        _, cdfG, distp = palm_competitive(Gvals.reshape(-1, 1), 'descend', True)
        # Unique values and corresponding modified ranks
        #cdfG, idx = np.unique(cdfG_raw, return_index=True)
        #distp = distp_raw.flatten()[idx] / Gvals.size
        # Sort back because unique sorts ascending
        #sort_idx = np.argsort(cdfG)[::-1]
        #cdfG = cdfG[sort_idx]
        #distp = distp[sort_idx]
        cdfG = np.unique(cdfG)
        distp = np.flipud(np.unique(distp)) / Gvals.size

        # Convert the data to p-values
        
        if G.size == 1:
            I = np.where(G < cdfG)[0]
            if I.size > 0:
                pvals = np.array([distp[I[0]]])
            else:
                pvals = np.array([0])
        else:
            pvals = np.zeros(G.size)
            for z in range(G.size):
                I = np.where(G[z] < cdfG)
                if I.size > 0:
                    pvals[z] = distp[I[0]]
            pvals = np.reshape(G.shape)
    
    return pvals

def palm_competitive(X, ord='ascend', mod=False):
    """
    Sort a set of values and return their competition
    ranks (standard 1224 or modified 1334).
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    orig_ord = ord
    if mod:
        if ord.lower() == 'ascend':
            ord = 'descend'
        elif ord.lower() == 'descend':
            ord = 'ascend'

    nR, nC = X.shape
    unsrtR = np.zeros(X.shape, dtype=np.float32)
    
    # Handle sorting direction
    if ord.lower() == 'ascend':
        tmp = np.argsort(X, axis=0)
    else:
        tmp = np.argsort(-X, axis=0)
        
    S = np.take_along_axis(X, tmp, axis=0)
    rev = np.argsort(tmp, axis=0)
    
    srtR = np.tile(np.arange(1, nR + 1).reshape(-1, 1), (1, nC)).astype(np.float32)
    
    for c in range(nC):
        col_S = S[:, c].copy()
        infpos = np.isinf(col_S) & (col_S > 0)
        infneg = np.isinf(col_S) & (col_S < 0)
        
        if np.all(infpos | infneg):
            raise ValueError("Data cannot be sorted. Maximum statistic is +Inf or -Inf for all permutations.")
            
        if np.any(infpos):
            col_S[infpos] = np.max(col_S[~infpos]) + 1
        if np.any(infneg):
            col_S[infneg] = np.min(col_S[~infneg]) - 1
            
        dd = np.diff(col_S)
        if np.any(np.isnan(dd)):
            raise ValueError("Data cannot be sorted. Check for NaNs or precision issues.")
            
        f = np.where(np.concatenate(([False], dd == 0)))[0]
        for pos in f:
            srtR[pos, c] = srtR[pos - 1, c]
            
        unsrtR[:, c] = srtR[rev[:, c], c]
        
        # Infinities are already handled via copying col_S, 
        # but we need to put them back in S if S is returned
        S[infpos, c] = np.inf
        S[infneg, c] = -np.inf

    if mod:
        unsrtR = nR - unsrtR + 1
        S = np.flipud(S)
        srtR = np.flipud(nR - srtR + 1)
        
    return unsrtR, S, srtR