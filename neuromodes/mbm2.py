
import numpy as np
import scipy.stats as sps

# import neuromodes.stats as nms
import mbm.palm
from neuromodes.basis import decompose

# ofc this willl be removed at some point
# jsut here to compare with the original fdr_bh
def fdr_bh(ps): 
    return sps.false_discovery_control(ps, method='bh')

# TODO CHECK RETURN_STAT FOR EVERYTHING. for one way and two way it is t. for anova it it p. etc. 
# TODO given how things are done in MBM (the permutations) consider removing two_sample. 
#   make it a subset of anova. can sqrt the f stat and then get the sign of the t stat.
def mbm_stat_map(
        inputMap, 
        designMatrix, 
        statTest, 
        # contrast=None, 
        # return_stat='t'
):
    
    # input map = (n_subjects, n_vertices)
    # design matrix = (n_subjects, n_predictors)
    # contrast = (n_predictors,) 

    if inputMap.ndim != 2:
        raise ValueError('inputMap must be a 2D array of shape (n_subjects, n_vertices).')
    n_subjects, n_vertices = inputMap.shape
    if designMatrix.ndim != 2:
        raise ValueError('designMatrix must be a 2D array of shape (n_subjects, n_predictors).')
    if designMatrix.shape[0] != n_subjects:
        raise ValueError('Number of rows in designMatrix must match number of subjects in inputMap.')
    n_predictors = designMatrix.shape[1]
    # if contrast is not None:
    #     if contrast.ndim != 1:
    #         raise ValueError('contrast must be a 1D array of shape (n_predictors,).')
    #     if contrast.shape[0] != n_predictors:
    #         raise ValueError('Length of contrast must match number of predictors in designMatrix.')

    # consider change ttest and anova to use glmw, same as ancova. 
    # more consistent. but better to use inbuilt where available?
    if statTest == 'one sample':
        if n_predictors != 1:
            raise ValueError('Design matrix must have exactly one predictor for one sample t-test.')
        statMap = sps.ttest_1samp(inputMap[designMatrix,:], 0, axis=0)[0]
    elif statTest == 'two sample':
        if n_predictors != 2:
            raise ValueError('Design matrix must have exactly two predictors for two sample t-test.')
        statMap = sps.ttest_ind(inputMap[designMatrix[:,0],:], inputMap[designMatrix[:,1],:], axis=0)[0]
    elif statTest == 'one way anova':
        # TODO : this is a p stat. need to clarify this across all options (return_stat not relevant)
        groups = [inputMap[designMatrix[:,i],:] for i in range(n_predictors)]
        statMap = sps.f_oneway(*groups, axis=0)[1]
    # elif statTest == 'ancova': # home made function, needs to be tested 2026/05/01
    #     statMap = nms.glmw(inputMap, designMatrix, 
    #                        w=np.ones(n_subjects), contrast=contrast, return_stat=return_stat)
    else:
        raise ValueError(f"Unsupported statTest: {statTest}")

    return statMap

# this function auto computes which tail we are in (auto two-tail?)
# should there be an option to specify the tail/make one tailed? (in this fn or another)?
# TODO : consider replacing palm_pareto with 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html
def mbm_estimate_p_val_tail(
        nullStat: np.ndarray,
        observedStat: float,
        pThr: float,
        stop=False
):
    # 1. Handle degenerate/empty data
    if np.isnan(observedStat) or np.all(np.isnan(nullStat)):
        return (1.0, False)
    
    if np.ptp(nullStat) == 0:
        # Distribution has no width; if observed is same as null, p=1, else p=0
        return (observedStat == nullStat[0], observedStat < nullStat[0])

    # 2. Determine which tail we are in
    # rev = True (Left tail/Negative), rev = False (Right tail/Positive)
    rev = observedStat < np.median(nullStat)

    # 3. Decision: To Pareto or not to Pareto?
    # We call palm_pareto if the observedStat is "Extreme" 
    # (even if it's beyond the current nullStat limits!)
    
    # Simple count-based p-value for reference
    if rev:
        count_p = np.mean(nullStat <= observedStat)
    else:
        count_p = np.mean(nullStat >= observedStat)

    # Only skip Pareto if the observation is very common (p > pThr)
    # If count_p is ~0 (beyond the limits), we DEFINITELY need Pareto.
    if count_p < pThr: 
        count_p = mbm.palm.palm_pareto(observedStat, nullStat, rev, pThr, stop)[0]
    
    pValueTail = 2 * count_p

    # if count_p > pThr:
    #     pValueTail = 2 * count_p
    # else:
    #     # This function handles the GPD fit for values beyond the limits
    #     pareto_res = mbm.palm.palm_pareto(observedStat, nullStat, rev, pThr, False)
    #     pValueTail = 2 * pareto_res[0]

    # Clean up output
    if isinstance(pValueTail, np.ndarray):
        pValueTail = pValueTail[0]
        
    return (float(np.clip(pValueTail, 0, 1)), rev)


def permutations_flip_sign(
        n_subjects: int,
        n_permutations: int,
        seed = None
): # output shape (n_subjects, n_permutations)
    rng = np.random.default_rng(seed)
    # so that adding more perms for the same seed doesn't change the previous perms
    return rng.choice(np.array([1, -1], dtype=np.int8), size=(n_permutations, n_subjects)).T

def permutations_shuffle_rows(
        n_subjects,  
        n_permutations,
        seed = None
): # output shape (n_subjects, n_permutations)
    rng = np.random.default_rng(seed)
    # so that adding more perms for the same seed doesn't change the previous perms
    data = rng.uniform(size=(n_permutations, n_subjects))
    out = np.argsort(data, axis=1).T
    return out


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
        statPThr = 0.05, # where the p value threshold for switching to pareto is & also the threshold for significant modes.
        statFDR = True, 
        n_modes = None, 
        n_permutations = 1000,
        seed = None
): 
    
    # Format / validate inputs
    n_vertices, n_subjects = maps.shape

    if n_modes is None:
        n_modes = emodes.shape[1]
    elif n_modes > emodes.shape[1]:
        raise ValueError('n_modes cannot be greater than the number of provided eigenmodes.')

    # 0 & 1
    # generate permutations
    # TODO look at speed options for this?
    # is it faster to avoid generating the full permuted maps
    # for one sample: can just flip the sign for each ii at the time it is needed?
    # for two sample can we just shuffle the design matrix 
    # if it is faster, consider using seedSequence to generate the seeds for each ii
    # then the input seed will be a 'master seed'. 
    # then generate n_permutations daughter seeds, need to all be different, and in reproducible order
    # see eigenstrapping for an example
    # maybe wont make a big difference as n_vertices >> n_subjects probably
    statMapNull = np.empty((n_vertices, n_permutations)) 
    if statTest == 'one sample': 
        perms = permutations_flip_sign(n_subjects, n_permutations, seed)
        for ii in range(n_permutations):
            permutedMaps = maps.T * perms[:, [ii]] # (n_subjects, n_vertices)
            statMapNull[:, ii] = mbm_stat_map(permutedMaps, statDesignMatrix, statTest).T # (n_vertices, 1)
    else: 
        perms = permutations_shuffle_rows(n_subjects, n_permutations, seed)
        for ii in range(n_permutations):
            permutedMaps = maps[:, perms[:, ii]].T # (n_subjects, n_vertices)
            statMapNull[:, ii] = mbm_stat_map(permutedMaps, statDesignMatrix, statTest).T # (n_vertices, 1)


    observedStatMap = mbm_stat_map(maps.T, statDesignMatrix, statTest).T # (n_vertices, 1)

    # 2
    # convert observed stat to p values using permutations
    # TODO make into a separate function (reuse for verts and modes)
    # TODO see if this can be vectorized across vertices
    # hmmm GPD is a single parameter fit (kinda - need to check piecewise-ness as param varies) 
    # so maybe the fitting can be done across all vertices simultaneously
    # then the p value estimation can be done across all vertices simultaneously
    # or consdier method of moments?
    permPMap = np.zeros(n_vertices, dtype=np.float32)
    permRevMap = np.zeros(n_vertices, dtype=np.bool_)
    for ii in range(n_vertices):
        permPMap[ii], permRevMap[ii] = mbm_estimate_p_val_tail(
            statMapNull[ii, :], observedStatMap[ii], statPThr, stop=False)
      
    if statFDR: 
        permPMap = fdr_bh(permPMap)


    # Eigenmode decomposition of observed and null stat maps
    # would decompose_kwargs be needed?
    eigBeta = decompose(observedStatMap, emodes=emodes, mass=mass, mode_counts=n_modes) # (n_modes, 1)
    betaNull = decompose(statMapNull, emodes=emodes, mass=mass, mode_counts=n_modes) # (n_modes, n_permutations)

    # 2
    eigPBeta = np.zeros(n_modes, dtype=np.float32)
    eigPRevBeta = np.zeros(n_modes, dtype=np.bool_)
    for ii in range(n_modes):
        eigPBeta[ii], eigPRevBeta[ii] = mbm_estimate_p_val_tail(
            betaNull[ii, :], eigBeta[ii], statPThr, stop=False)
    
    if statFDR:
        eigPBeta = fdr_bh(eigPBeta)

    # Take those above results, extract significant modes, and reconstruct the stat map
    reconMap = emodes @ (eigBeta * (eigPBeta < statPThr))






