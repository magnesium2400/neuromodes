
import numpy as np
import scipy.stats as sps

# import neuromodes.stats as nms
import neuromodes.palm
import neuromodes.glmw as glmw
from neuromodes.basis import decompose

# TODO rename all variables to be more descriptive

# ofc this willl be removed at some point
# jsut here to compare with the original fdr_bh
def fdr_bh(ps): 
    return sps.false_discovery_control(ps, method='bh')

def _stat_test_to_contrast(statTest, n_predictors):
    if statTest == 'one sample':
        if n_predictors != 1:
            raise ValueError('Design matrix must have exactly one predictor for one sample t-test.')
        return np.array([[1.0]])
        
    elif statTest == 'two sample':
        if n_predictors != 2:
            raise ValueError('Design matrix must have exactly two predictors for two sample t-test.')
        return np.array([[1.0, -1.0]])
        
    elif statTest == 'one way anova':
        if n_predictors < 2:
            raise ValueError('Design matrix must have 2 or more group predictors for an ANOVA.')
        # creates a matrix like [[1, -1, 0], [1, 0, -1]] for 3 groups
        return np.hstack([np.ones((n_predictors - 1, 1)), -np.eye(n_predictors - 1)])
    
    elif statTest == 'ancova':
        raise ValueError("Contrast must be explicitly provided for ANCOVA tests.")
        
    else:
        raise ValueError(f"Unsupported statTest: {statTest}")

# TODO replace with yield
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
    choices = np.array([1, -1], dtype=np.int8)
    for i, s in enumerate(child_seeds):
        out[:, i] = np.random.default_rng(s).choice(choices, size=n_subjects)
        
    return out

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
    # int32 to save memory (int64 is overkill unless > 2 billion subjects)
    out = np.empty((n_subjects, n_permutations), dtype=np.int32, order='F')
    
    # 3. Fill the array column-by-column directly
    for i, s in enumerate(child_seeds):
        out[:, i] = np.random.default_rng(s).permutation(n_subjects)
        
    return out


# TODO look at speed options for this?
# is it faster to avoid generating the full permuted maps
# for one sample: can just flip the sign for each ii at the time it is needed?
# for two sample can we just shuffle the design matrix 
# maybe also put the if statement inside the loop if this is the case
# if it is faster, consider using seedSequence to generate the seeds for each ii
# then the input seed will be a 'master seed'. 
# then generate n_permutations daughter seeds, need to all be different, and in reproducible order
# see eigenstrapping for an example
# maybe wont make a big difference as n_vertices >> n_subjects probably
# TODO investigate user inputting a perms matrix (n_subjects, n_permutations) 
# TODO have a way to turn off the permutations (only do observed)
# TODO consider outputting perms
# TODO create ?helper functions that use yield ?instead of generating whole perm mat
def mbm_generate_stat_map_nulls(
        maps, # (n_vertices, n_subjects)
        statDesignMatrix,   # (n_subjects,) or (n_subjects, n_predictors) depending on statTest
        contrastMatrix, # (n_contrasts, n_predictors)
        massMatrix, 
        n_permutations = 1000,
        seed = None
):
    n_vertices, n_subjects = maps.shape
    
    statMapNull = np.empty((n_vertices, n_permutations)) 
    if contrastMatrix.size == 1: # one sample t test
        perms = permutations_flip_sign(n_subjects, n_permutations, seed)
        for ii in range(n_permutations):
            permutedMaps = maps.T * perms[:, [ii]] # (n_subjects, n_vertices)
            statMapNull[:, ii] = glmw.glmw_test(permutedMaps, statDesignMatrix, massMatrix, contrastMatrix)[0].T # (n_vertices, 1)
    else: 
        perms = permutations_shuffle_rows(n_subjects, n_permutations, seed)
        for ii in range(n_permutations):
            permutedMaps = maps[:, perms[:, ii]].T # (n_subjects, n_vertices)
            statMapNull[:, ii] = glmw.glmw_test(permutedMaps, statDesignMatrix, massMatrix, contrastMatrix)[0].T # (n_vertices, 1)

    return statMapNull


# TODO make into a separate function (reuse for verts and modes)
# TODO see if this can be vectorized across vertices
# hmmm GPD is a single parameter fit (kinda - need to check piecewise-ness as param varies) 
# so maybe the fitting can be done across all vertices simultaneously
# then the p value estimation can be done across all vertices simultaneously
# or consider method of moments?
# TODO at the very least, consider finding naive p val for all vertices first, 
# then only doing the GPD fit for those that are below the threshold.
def mbm_calc_p_vals(
        observedStatMap, # (n_vertices,)
        statMapNull, # (n_vertices, n_permutations)
        statPThr = 0.05, # threshold for tail extrapolation with pareto
        alternative = 'two-sided', # 'two-sided', 'greater', 'less' # TODO validate here
        stop = False
): 

    if observedStatMap.ndim != 1:
        raise ValueError(f"observedStatMap must be 1D, got shape {observedStatMap.shape}")

    # non parametric p values from permutations
    # rev = True (Left tail/Negative), rev = False (Right tail/Positive)    
    permPMap = np.mean(statMapNull >= observedStatMap[:, None], axis=1)
    if alternative == 'greater':
        permRevMap = np.zeros_like(observedStatMap, dtype=bool)
    elif alternative == 'two-sided':
        permPMap = 2 * np.minimum(permPMap, 1 - permPMap)
        permRevMap = observedStatMap < np.median(statMapNull, axis=1)
    elif alternative == 'less':
        permPMap = 1 - permPMap
        permRevMap = np.ones_like(observedStatMap, dtype=bool)
    else:
        raise ValueError(f"Unsupported alternative hypothesis: {alternative}")
    
    for ii in range(observedStatMap.shape[0]):
        if permPMap[ii] < statPThr: # only do the pareto fit if we are in the tail
            p, _, _, _ = neuromodes.palm.palm_pareto(
                observedStatMap[ii], 
                statMapNull[ii, :], 
                permRevMap[ii], 
                statPThr, 
                stop)
            permPMap[ii] = 2 * p[0]

    permPMap = np.clip(permPMap, 0, 1)
    return permPMap, permRevMap



# TODO : generate smaller single responsibility functions 
# ?0. generate permutations (n_subjects, n_permutations). have to see if it is worthwhile 
# saving all these in memory at the same time for large number of permutations 
# (other option, generate on the fly, perhaps slower but less memory)  
# 1. generate statMapNull (n_vertices, n_permutations)
# 2. finding p values for A (observed stat/beta map) vs B (null stat/beta maps)
#   A is (n_vertices/n_modes, 1), collapsed from (n_vertices/n_modes, n_subjects) using the stat test
#   B is (n_vertices/n_modes, n_permutations)
#   users can FDR correct easily themselves if desired, so just return the uncorrected p values
def mbm_example_workflow(
        maps, # (n_vertices, n_subjects)
        emodes, # (n_vertices, n_modes)
        mass, # (n_vertices, n_vertices)
        statTest,           # cant do ancova as there is no contrast
        statDesignMatrix,   # (n_subjects,) or (n_subjects, n_predictors) depending on statTest
        statPThr = 0.05, # consider making two thresh? for tail estimation and for mode significance?
        statFDR = True, 
        n_modes = None, 
        n_permutations = 1000,
        seed = None
): 
    
    # 0 & 1
    # generate permutations
    ident = np.eye(maps.shape[1])
    contrastMatrix = _stat_test_to_contrast(statTest, statDesignMatrix.shape[1])

    observedStatMap, _ = glmw.glmw_test(
        maps.T, statDesignMatrix, ident, contrastMatrix
    ) # (n_vertices, 1)
    nullStatMaps = mbm_generate_stat_map_nulls(
        maps, statDesignMatrix, contrastMatrix, ident, n_permutations, seed
    )

    # 2
    # convert observed stat to p values using tail estimation
    permPMap, permRevMap = mbm_calc_p_vals(observedStatMap, nullStatMaps, statPThr)
    if statFDR:
        permPMap = fdr_bh(permPMap)
    sig_stat_mask = permPMap < statPThr
    recon_stat_map = observedStatMap * sig_stat_mask


    # Eigenmode decomposition of observed and null stat maps
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

