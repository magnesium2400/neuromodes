
import numpy as np
import scipy.stats as sps
from warnings import warn
from collections.abc import Iterable
from typing import Literal

# import neuromodes.stats as nms
# import neuromodes.palm
import neuromodes.glmw as glmw
from neuromodes.basis import decompose
from neuromodes.palm import andersondarling

# TODO rename all variables to be more descriptive
# TODO remove mbm_ from function names

# ofc this willl be removed at some point
# jsut here to compare with the original fdr_bh
def fdr_bh(ps): 
    return sps.false_discovery_control(ps, method='bh')

# TODO replace with yield?
def permutations_flip_sign(
    n_subjects: int,  
    n_permutations: int,
    seed: int | None = None
) -> np.ndarray:
    """
    Generates independent random sign flips (1 or -1) for subjects.
    
    Uses spawned SeedSequences to guarantee that generating additional
    permutations for a given seed will not alter the streams of previously
    generated permutations.

    Args:
        n_subjects: The number of subjects (rows).
        n_permutations: The number of permutations (columns).
        seed: An integer or np.random.SeedSequence for reproducibility.

    Returns:
        np.ndarray: An array of shape (n_subjects, n_permutations) of type int8.
    """
    # 1. Initialize the seeds
    child_seeds = np.random.SeedSequence(seed).spawn(n_permutations)
    
    # 2. Pre-allocate the output array (F-contiguous for column-wise writing)
    out = np.empty((n_subjects, n_permutations), dtype=np.int8, order='F')
    
    # 3. Fill the array column-by-column directly
    for i, s in enumerate(child_seeds):
        out[:, i] = np.random.default_rng(s).integers(0, 2, size=n_subjects, dtype=np.int8) * 2 - 1
        
    return out

# TODO consider making this faster/better/more pythonic with 
# yield/Lazy evalutions/no need to predetermine number of permutations?
def permutations_shuffle_rows(
    n_subjects: int,  
    n_permutations: int,
    seed: int | None = None
) -> np.ndarray:
    """
    Generates independent random permutations of subject indices.
    
    Uses spawned SeedSequences to guarantee that generating additional
    permutations for a given seed will not alter the streams of previously
    generated permutations.

    Args:
        n_subjects: The number of subjects (rows).
        n_permutations: The number of permutations (columns).
        seed: An integer or np.random.SeedSequence for reproducibility.

    Returns:
        np.ndarray: An array of shape (n_subjects, n_permutations) of type int32.
    """
    # 1. Initialize the seeds
    child_seeds = np.random.SeedSequence(seed).spawn(n_permutations)
    
    # 2. Pre-allocate the output array (F-contiguous for column-wise writing)
    out = np.empty((n_subjects, n_permutations), dtype=np.min_scalar_type(n_subjects), order='F')
    
    # 3. Fill the array column-by-column directly
    for i, s in enumerate(child_seeds):
        out[:, i] = np.random.default_rng(s).permutation(n_subjects)
        
    return out

# TODO add tests if keeping this as a standalone function
def mbm_generate_permutations(
        response, # (n_subjects, n_data)
        permutationMethod, 
        n_permutations,
        seed = None
):

    # Return a function that generates the permuted response for a given permutation index ii
    # permutedMaps(ii) will be of shape (n_subjects, n_data)

    n_subjects = response.shape[0]

    if permutationMethod == 'flip_sign':
        perms = permutations_flip_sign(n_subjects, n_permutations, seed) # (n_subjects, n_permutations)
        permutedMaps = lambda ii: response * perms[:, ii][:, None]
    elif permutationMethod == 'shuffle_rows': 
        perms = permutations_shuffle_rows(n_subjects, n_permutations, seed) # (n_subjects, n_permutations)
        permutedMaps = lambda ii: response[perms[:, ii], :]
    else:
        raise ValueError(f"Unsupported permutation method: {permutationMethod}")

    return permutedMaps, perms


# Currently only supports one ttest at a time (1 contrast). For multiple contrasts, use ftestw/ANOVA.
# doing multiple ttests concurrently is dangerous in terms of multiple comparisons, so we don't want to encourage it.
# If you want to do multiple ttests, they have to be run separately and then corrected for multiple comparisons (e.g. FDR).
# If you want to see this functionality added, please open an issue on GitHub and we can discuss the best way to implement it.
# It would probably involve different tests (contrasts) being stacked along axis=2
def _mbm_generate_stat_map(
        design, # (n_subjects, n_predictors)
        response, # (n_subjects, n_data)
        contrast, # (n_contrasts, n_predictors)
        statTest, # TODO consider replace with t, f, z. Enum?
):
    
    ident = np.eye(response.shape[0]) # TODO change to scipy sparse?

    # TODO consider replacing glmw with inbuilt functions
    # TODO consider creating non-weighted functions that just called the weighted version with ident
    if statTest == 'ttest':
        statMap = glmw.ttestw(design, response, contrast, ident).statistic
    elif statTest == 'ztest':
        statMap = glmw.ztestw(design, response, contrast, ident).statistic
    elif statTest == 'ftest':
        statMap = glmw.ftestw(design, response, contrast, ident)[0]
    else:
        raise ValueError(f"Unsupported stat test: {statTest}")

    return statMap # shape (n_data,)


# TODO look at speed options for this?
# is it faster to avoid generating the full permuted response
# for one sample: can just flip the sign for each ii at the time it is needed?
# for two sample can we just shuffle the design matrix 
# maybe also put the if statement inside the loop if this is the case
# if it is faster, consider using seedSequence to generate the seeds for each ii
# then the input seed will be a 'master seed'. 
# then generate n_permutations daughter seeds, need to all be different, and in reproducible order
# see eigenstrapping for an example
# TODO future: stat testing support 3d data so that this can be done without a loop
# TODO rename this function or the previous one
def mbm_generate_stat_maps(
        design, # (n_subjects, n_predictors)
        response, # (n_subjects, n_data)
        contrast, # (n_contrasts, n_predictors)
        statTest,
        permutationMethod = 'shuffle_rows', # 'flip_sign', 'shuffle_rows', None
        n_permutations = 0, # TODO change this!! boolean, None, separate function name, ??
        seed = None
):

    if design.ndim != 2:
        raise ValueError(f"Design matrix must be 2D. Received shape: {design.shape}")
    if response.ndim != 2:
        raise ValueError(f"Response matrix must be 2D. Received shape: {response.shape}")
    if contrast.ndim != 2:
        raise ValueError(f"Contrast matrix must be 2D. Received shape: {contrast.shape}")

    if n_permutations == 0: 
        if seed is not None:
            warn("seed is ignored when permutationMethod is 0.")
        return _mbm_generate_stat_map(design, response, contrast, statTest)

    # Permutation only applies to the columns of the design matrix that are involved in the contrast
    # Don't permute the nuisance predictors
    targetCols = np.any(contrast != 0, axis=0)

    permutedDesigns, _ = mbm_generate_permutations(
        design[:, targetCols], permutationMethod, n_permutations, seed
    )

    # Iterate over number of permutations
    statMapNull = np.empty((response.shape[1], n_permutations))
    for ii in range(n_permutations):
        curr = design.copy()
        curr[:, targetCols] = permutedDesigns(ii)
        statMapNull[:, ii] = _mbm_generate_stat_map(
            curr, response, contrast, statTest
        )

    return statMapNull # shape (n_data, n_permutations)


# TODO FUTURE look at ways of vectorising this across vertices ?Newton method for MLE ?method of moments
# TODO consider returning the fit vals/threshold/ad test for each vertex as well
# TODO FUTURE support observed_stat_map being 2D for alternative == two-sided ?recursion
def mbm_calc_p_vals(
        # TODO change names!
        observedStatMap, # (n_data,) or (n_data, n_obs) if alternative != 'two-sided'
        statMapNull, # (n_data, n_permutations)
        alternative: Literal['two-sided', 'greater', 'less'] = 'two-sided', 
        n_for_pval_recalculation: int | None = 9, # None to disable recalculation, 0 to only do something past all the nulls, 9 to match Knijenberg, n_permutations to do all
        n_for_tail_estimation: int | Iterable[int] = tuple(range(250, 2, -10)), # match Knijenburg
        gof_acceptance_level: float | None = 0.05, # match Knijenburg but perhaps 0.10 is better?
        return_tail = False # TODO add overloads
): 

    # Note that n_for_tail_estimation and n_for_pval_recalculation should be specified as 
    # integers (like in Knijenberg et al. 2009), NOT as fractions (like in PALM).

    # 0. Input validation
    if is_vector_obs := observedStatMap.ndim == 1:
        observedStatMap = observedStatMap[:, None] # reshape to (n_data, 1) for consistent processing
    elif observedStatMap.ndim == 2:
        if alternative == 'two-sided':
            # If you would like this to be implemented, please leave a GitHub issue and we can have a look
            raise NotImplementedError("observedStatMap cannot be 2D for two-sided alternative. Please provide a 1D array of observed statistics.")
    else:
        raise ValueError(f"observedStatMap must be 1D or 2D, got shape {observedStatMap.shape}")

    if statMapNull.ndim != 2:
        raise ValueError(f"statMapNull must be 2D, got shape {statMapNull.shape}")
    if statMapNull.shape[0] != observedStatMap.shape[0]:
        raise ValueError(f"statMapNull and observedStatMap must have the same number of rows. Got {statMapNull.shape[0]} and {observedStatMap.shape[0]}.")
    n_permutations = statMapNull.shape[1]

    if isinstance(n_for_tail_estimation, int):
        n_for_tail_estimation = (n_for_tail_estimation,)
    if not isinstance(n_for_tail_estimation, Iterable):
        raise TypeError(f"n_for_tail_estimation must be an integer or iterable, got {type(n_for_tail_estimation)}")
    for n in n_for_tail_estimation:
        if not isinstance(n, int):
            raise TypeError(f"All values in n_for_tail_estimation must be integers, got {type(n)}")
        if n > n_permutations:
            warn(f"Some values in n_for_tail_estimation are greater than n_permutations ({n_permutations}). These will be ignored.")
        if n < 2:
            warn(f"Some values in n_for_tail_estimation are less than 2. These will be ignored.")
    nte_copy = [nte for nte in n_for_tail_estimation if 2 <= nte <= n_permutations]
    if n_for_tail_estimation and not nte_copy:
        warn(f"All values in n_for_tail_estimation ({n_for_tail_estimation}) are invalid. No p-values will be recalculated.")

    if n_for_pval_recalculation is not None:
        if n_for_pval_recalculation < 0:
            warn(f"n_for_pval_recalculation ({n_for_pval_recalculation}) is less than 0. No p-values will be recalculated.")
        if n_for_pval_recalculation > n_permutations: # Don't warn for n_permutations (allow user to force recalculation))
            warn(f"n_for_pval_recalculation ({n_for_pval_recalculation}) is greater than n_permutations ({n_permutations}). All p-values will be recalculated.")
    
    # 1. Get naive p values (P'_ecdf) and setup for calculation of P_gpd
    tail_counts, tail_dir = _count_tails(observedStatMap, statMapNull, alternative)
    mask_2d = (tail_counts <= n_for_pval_recalculation
            if n_for_pval_recalculation is not None
            else np.zeros_like(tail_counts, dtype=bool))
    mask_1d = np.any(mask_2d, axis=1)

    # Orient observed statistics and null distributions so all tails act as upper/right tails
    factor = 1 - 2 * tail_dir[mask_1d].astype(int) # (n_recalc, n_obs)
    obs = observedStatMap[mask_1d] * factor  # (n_recalc, n_obs)
    nulls_nondesc = np.sort(statMapNull[mask_1d, :] * factor[:,:1], axis=1)  # (n_recalc, n_permutations)

    # 2. Main loop: iterate over each row that needs recalculation
    # Try different numbers of tail elements for GPD fitting, until AD test passes
    for ob, nulls_row, mask_row, ii in zip(obs, nulls_nondesc, mask_2d[mask_1d], np.where(mask_1d)[0]):
        # TODO FUTURE get unique values of nulls_row[-nte_copy] in stable order
        # as we only need to test each unique threshold
        for nte in nte_copy: 
            # TODO FUTURE investigate whether Knijnenburg midpoint threshold or exact threshold is better
            # Note that if using the exact threshold, the zero elements can (?should) be removed from AD test

            # Find the threshold and the exceedances for the current tail size estimation
            t, excess_row = _get_threshold_and_exceedances(nulls_row, nte)
            if np.var(excess_row) == 0.0:
                if nte == np.max(nte_copy):
                    warn(f"All tail values are identical for vertex {ii}.")
                    break
                else:
                    continue

            # this will only be used after the fitting, but we cimpute it early in case we need to early exit
            ob_mask = mask_row & (ob > t) 
            if not np.any(ob_mask):
                if nte == np.max(nte_copy):
                    warn(f"Observed statistic {ob} is below the minimum threshold {t} for vertex {ii}.")
                    break
                else: 
                    continue

            # Fit GPD with location fixed to threshold t
            # Could use try-catch to continue to other nte/vertices if one fit errors out
            # TODO FUTURE: consider how to return these fit values and AD stats/result to user
            fit, gofp = _fit_and_test_exceedances(excess_row, gof_acceptance_level)
            if fit[0] <= -0.5 or fit[0] > 1.0 or fit[2] <= 0.0: # shape parameter k <= -0.5 or scale parameter a <= 0.0
                continue
            if gof_acceptance_level is not None and gofp <= gof_acceptance_level: # next iter if fit is poor
                continue

            tail_counts[ii, ob_mask] = excess_row.size * sps.genpareto.sf(ob[ob_mask] - t, *fit)
            mask_row[ob_mask] = False # mark as successfully recalculated
            if not np.any(mask_row): # if all recalculated, break out of nte loop
                break

        if np.any(mask_row):
            warn(f"GPD fitting failed for vertex {ii}. Using non-parametric p-value.")

    # 3. Output
    output = [_counts_to_pvals(tail_counts, n_permutations, alternative)]
    if return_tail:
        output.append(tail_dir)
    # TODO FUTURE if return_fit: output.append(...)
    # TODO FUTURE if return_ad: output.append(...)

    if is_vector_obs:
        output = [np.squeeze(arr, axis=1) for arr in output]

    return tuple(output) if len(output) > 1 else output[0]


def _count_tails(obs_2d, nulls_2d, alternative): 
    obs_3d = obs_2d[:, np.newaxis, :]
    null_3d = nulls_2d[:, :, np.newaxis]
    geq = np.sum(null_3d >= obs_3d, axis=1).astype(float)
    leq = np.sum(null_3d <= obs_3d, axis=1).astype(float)
    if alternative == 'greater':
        tail_counts = geq.copy()
        tail_dir = np.zeros_like(geq, dtype=bool)
    elif alternative == 'less':
        tail_counts = leq.copy()
        tail_dir = np.ones_like(leq, dtype=bool)
    elif alternative == 'two-sided':
        tail_counts = np.minimum(geq, leq).copy() # will be doubled at the end
        tail_dir = leq < geq
    else:
        raise ValueError(f"Unsupported alternative hypothesis: {alternative}")
    return tail_counts, tail_dir

def _counts_to_pvals(pvals, n_permutations, alternative):
    if alternative == 'two-sided':
        pvals = 2 * pvals
    return np.clip(pvals / n_permutations, 0.0, 1.0)

def _naive_pvals(obs_2d, nulls_2d, alternative):
    tail_counts, tail_dir = _count_tails(obs_2d, nulls_2d, alternative)
    return _counts_to_pvals(tail_counts, nulls_2d.shape[1], alternative), tail_dir

def _get_threshold_and_exceedances(nulls_nondesc, n_tail): 
    tidx = np.searchsorted(nulls_nondesc, nulls_nondesc[-n_tail], side='left')
    if tidx != 0: 
        t = (nulls_nondesc[tidx-1] + nulls_nondesc[tidx]) / 2.0
    else: # tidx = 0: take a half step before the first value
        t = (3 * nulls_nondesc[0] - nulls_nondesc[1]) / 2.0
    excess_row = nulls_nondesc[tidx:] - t # same as nulls_nondesc[nulls_nondesc > t] 
    return t, excess_row

# This is here mostly to facilitate testing
def _fit_and_test_exceedances(excess_row, gof_acceptance_level):
    fit = sps.genpareto.fit(excess_row, floc=0.0)
    if gof_acceptance_level is not None: 
        gofp = andersondarling(
            np.sort(sps.genpareto.cdf(excess_row, *fit)), 
            k=-fit[0])[0] # Choulakian & Stephens parameterize shape parameter k = -c
    else: 
        gofp = np.nan # gof = (1.0, 0.0) # could instead use dummy values with p = 1 and a2 = 0
    return fit, gofp






# TODO : generate smaller single responsibility functions 
# ?0. generate permutations (n_subjects, n_permutations). have to see if it is worthwhile 
# saving all these in memory at the same time for large number of permutations 
# (other option, generate on the fly, perhaps slower but less memory)  
# 1. generate statMapNull (n_data, n_permutations)
# 2. finding p values for A (observed stat/beta map) vs B (null stat/beta response)
#   A is (n_data/n_modes, 1), collapsed from (n_data/n_modes, n_subjects) using the stat test
#   B is (n_data/n_modes, n_permutations)
#   users can FDR correct easily themselves if desired, so just return the uncorrected p values
def mbm_example_workflow(
        response, # (n_data, n_subjects)
        emodes, # (n_data, n_modes)
        mass, # (n_data, n_data)
        statTest,           # cant do ancova as there is no contrast
        design,   # (n_subjects,) or (n_subjects, n_predictors) depending on statTest
        statPThr = 0.05, # consider making two thresh? for tail estimation and for mode significance?
        statFDR = True, 
        n_modes = None, 
        n_permutations = 1000,
        seed = None
): 
    
    # 0 & 1
    # generate permutations
    ident = np.eye(response.shape[1])
    contrast = _stat_test_to_contrast(statTest, design.shape[1])

    observedStatMap, _ = glmw.glmw_test(
        response.T, design, ident, contrast
    ) # (n_data, 1)
    nullStatMaps = mbm_generate_stat_map_nulls(
        response, design, contrast, ident, n_permutations, seed
    )

    # 2
    # convert observed stat to p values using tail estimation
    permPMap, permRevMap = mbm_calc_p_vals(observedStatMap, nullStatMaps, statPThr)
    if statFDR:
        permPMap = fdr_bh(permPMap)
    sig_stat_mask = permPMap < statPThr
    recon_stat_map = observedStatMap * sig_stat_mask


    # Eigenmode decomposition of observed and null stat response
    # would decompose_kwargs be needed?
    # TODO update to new decompose
    eigBeta = decompose(observedStatMap, emodes=emodes, mass=mass) # (n_modes, 1)
    betaNull = decompose(nullStatMaps, emodes=emodes, mass=mass) # (n_modes, n_permutations)

    # 2
    eigBeta = eigBeta.flatten() # TODO remove this
    eigPBeta, eigPRevBeta = mbm_calc_p_vals(eigBeta, betaNull, statPThr)    
    if statFDR:
        eigPBeta = fdr_bh(eigPBeta)

    # Take those above results, extract significant modes, and reconstruct the stat map
    sig_mode_mask = eigPBeta < statPThr
    sig_mode_indices = np.where(sig_mode_mask)[0]
    recon_mode_map = emodes @ (eigBeta * sig_mode_mask)


    return recon_mode_map, sig_mode_indices
