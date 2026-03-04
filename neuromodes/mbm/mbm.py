#!/usr/bin/env python3

"""
Module for MBM.
"""

import neuromodes.basis
import palm
import os
import nibabel
import numpy
import scipy.io
import scipy.stats

def fdr_bh(pvals, q=0.05, method='pdep', report='no'):
    """
    % fdr_bh() - Executes the Benjamini & Hochberg (1995) and the Benjamini &
    %            Yekutieli (2001) procedure for controlling the false discovery 
    %            rate (FDR) of a family of hypothesis tests. FDR is the expected
    %            proportion of rejected hypotheses that are mistakenly rejected 
    %            (i.e., the null hypothesis is actually true for those tests). 
    %            FDR is a somewhat less conservative/more powerful method for 
    %            correcting for multiple comparisons than procedures like Bonferroni
    %            correction that provide strong control of the family-wise
    %            error rate (i.e., the probability that one or more null
    %            hypotheses are mistakenly rejected).
    %
    %            This function also returns the false coverage-statement rate 
    %            (FCR)-adjusted selected confidence interval coverage (i.e.,
    %            the coverage needed to construct multiple comparison corrected
    %            confidence intervals that correspond to the FDR-adjusted p-values).
    %
    %
    % Usage:
    %  >> [h, crit_p, adj_ci_cvrg, adj_p]=fdr_bh(pvals,q,method,report);
    %
    % Required Input:
    %   pvals - A vector or matrix (two dimensions or more) containing the
    %           p-value of each individual test in a family of tests.
    %
    % Optional Inputs:
    %   q       - The desired false discovery rate. {default: 0.05}
    %   method  - ['pdep' or 'dep'] If 'pdep,' the original Bejnamini & Hochberg
    %             FDR procedure is used, which is guaranteed to be accurate if
    %             the individual tests are independent or positively dependent
    %             (e.g., Gaussian variables that are positively correlated or
    %             independent).  If 'dep,' the FDR procedure
    %             described in Benjamini & Yekutieli (2001) that is guaranteed
    %             to be accurate for any test dependency structure (e.g.,
    %             Gaussian variables with any covariance matrix) is used. 'dep'
    %             is always appropriate to use but is less powerful than 'pdep.'
    %             {default: 'pdep'}
    %   report  - ['yes' or 'no'] If 'yes', a brief summary of FDR results are
    %             output to the MATLAB command line {default: 'no'}
    %
    %
    % Outputs:
    %   h       - A binary vector or matrix of the same size as the input "pvals."
    %             If the ith element of h is 1, then the test that produced the 
    %             ith p-value in pvals is significant (i.e., the null hypothesis
    %             of the test is rejected).
    %   crit_p  - All uncorrected p-values less than or equal to crit_p are 
    %             significant (i.e., their null hypotheses are rejected).  If 
    %             no p-values are significant, crit_p=0.
    %   adj_ci_cvrg - The FCR-adjusted BH- or BY-selected 
    %             confidence interval coverage. For any p-values that 
    %             are significant after FDR adjustment, this gives you the
    %             proportion of coverage (e.g., 0.99) you should use when generating
    %             confidence intervals for those parameters. In other words,
    %             this allows you to correct your confidence intervals for
    %             multiple comparisons. You can NOT obtain confidence intervals 
    %             for non-significant p-values. The adjusted confidence intervals
    %             guarantee that the expected FCR is less than or equal to q
    %             if using the appropriate FDR control algorithm for the  
    %             dependency structure of your data (Benjamini & Yekutieli, 2005).
    %             FCR (i.e., false coverage-statement rate) is the proportion 
    %             of confidence intervals you construct
    %             that miss the true value of the parameter. adj_ci=NaN if no
    %             p-values are significant after adjustment.
    %   adj_p   - All adjusted p-values less than or equal to q are significant
    %             (i.e., their null hypotheses are rejected). Note, adjusted 
    %             p-values can be greater than 1.
    """

    pvals = numpy.array(pvals)
    
    # Check for p-value validity
    if pvals.size == 0:
        raise ValueError('You need to provide a vector or matrix of p-values.')
    else:
        if numpy.any(pvals < 0):
            raise ValueError('Some p-values are less than 0.')
        elif numpy.any(pvals > 1):
            raise ValueError('Some p-values are greater than 1.')

    s = pvals.shape
    # Flatten pvals to 1D for processing
    p_flat = pvals.ravel()
    m = len(p_flat) # number of tests

    # Sort p-values
    sort_ids = numpy.argsort(p_flat)
    p_sorted = p_flat[sort_ids]
    
    # indexes to return p_sorted to pvals order
    unsort_ids = numpy.argsort(sort_ids)

    indices = numpy.arange(1, m + 1)
    if method.lower() == 'pdep':
        # BH procedure for independence or positive dependence
        thresh = indices * q / m
        wtd_p = m * p_sorted / indices
        
    elif method.lower() == 'dep':
        # BH procedure for any dependency structure
        denom = m * numpy.sum(1.0 / indices)
        thresh = indices * q / denom
        wtd_p = denom * p_sorted / indices
        # Note, it can produce adjusted p-values greater than 1!
        
    else:
        raise ValueError("Argument 'method' needs to be 'pdep' or 'dep'.")

    # compute adjusted p-values; This can be a bit computationally intensive
    adj_p_flat = numpy.full(m, numpy.nan)
    wtd_p_sindex = numpy.argsort(wtd_p)
    wtd_p_sorted = wtd_p[wtd_p_sindex]
    
    nextfill = 0
    for k in range(m):
        if wtd_p_sindex[k] >= nextfill:
            # Fill from nextfill up to the original sorted index of the k-th smallest weighted p-value
            adj_p_flat[nextfill : wtd_p_sindex[k] + 1] = wtd_p_sorted[k]
            nextfill = wtd_p_sindex[k] + 1
            if nextfill >= m:
                break
    
    # Map back to original order and reshape
    adj_p = adj_p_flat[unsort_ids].reshape(s)

    rej = p_sorted <= thresh
    max_ids = numpy.where(rej)[0]
    
    if max_ids.size == 0:
        crit_p = 0
        h = numpy.zeros_like(pvals, dtype=int)
        adj_ci_cvrg = numpy.nan
    else:
        max_id = max_ids[-1] # find greatest significant pvalue
        crit_p = p_sorted[max_id]
        h = (pvals <= crit_p).astype(int)
        adj_ci_cvrg = 1 - thresh[max_id]

    if report.lower() == 'yes':
        n_sig = numpy.sum(p_sorted <= crit_p)
        if n_sig == 1:
            print('Out of %d tests, %d is significant using a false discovery rate of %f.' % (m, n_sig, q))
        else:
            print('Out of %d tests, %d are significant using a false discovery rate of %f.' % (m, n_sig, q))
            
        if method.lower() == 'pdep':
            print('FDR/FCR procedure used is guaranteed valid for independent or positively dependent tests.')
        else:
            print('FDR/FCR procedure used is guaranteed valid for independent or dependent tests.')

    return h, crit_p, adj_ci_cvrg, adj_p


def mbm_estimate_p_val_tail(
        nullStat: numpy.ndarray,
        observedStat: float,
        pThr: float

):
    # dimension and type checking

    #corr_mat1out = False

    if numpy.isnan(observedStat) and numpy.all(numpy.isnan(nullStat)):
        # no tail
        pValueTail = 1
        rev = True
    elif numpy.max(nullStat) <= observedStat:
        # right tail
        pValueTail = 0
        rev = False
    elif numpy.min(nullStat) >= observedStat:
        # left tail
        pValueTail = 0
        rev = True
    elif numpy.ptp(nullStat) == 0:
        # left tail
        pValueTail = 1
        rev = True
    else:
        rev = observedStat < numpy.median(nullStat)
        pValueTail = 2 * palm.palm_pareto(observedStat, nullStat, rev, pThr, False)[0]
        if isinstance(pValueTail, numpy.ndarray):
            pValueTail = pValueTail[0]

    
    return (float(pValueTail), rev)


def mbm_stat_map(
        inputMap: numpy.ndarray,
        designMatrix: numpy.ndarray,
        statTest: str
):
    
    # dimension and type checking
    #[nSub,nVertice] = size(y);
    numSubjects = inputMap.shape[0]
    numVertices = inputMap.shape[1]

    if statTest.lower() == "one sample":
        statMap = scipy.stats.ttest_1samp(numpy.compress(designMatrix == 1, inputMap, axis=0), 0)[0]
    elif statTest.lower() == "two sample":
        statMap = scipy.stats.ttest_ind(
            numpy.compress(designMatrix[:, 0] == 1, inputMap, axis=0),
            numpy.compress(designMatrix[:, 1] == 1, inputMap, axis=0)
        )[0]
        
        statMap[numpy.isnan(statMap)] = 0

    elif statTest.lower() == "one way anova":
        pass
    elif statTest.lower() == "ancova_f":
        pass
    elif statTest.lower() == "ancova_p":
        pass
    else:
        raise ValueError('Unsupported statistical test')
    return statMap

#@profile
def mbm(
        anatListFile: str,
        anatMaskFile: str,
        statTest: str,
        statDesignFile: str,
        statNPer: int,
        statPThr: float,
        statThresh: float,
        statFDR: bool,
        eigFile: str,
        massFile: str,
        nEigenmodes: int
):
    
    # load the anatList
    # mbm_read_inputs, I will only do the giftis since this isnt required in the neurodmodes code
    if not os.path.isfile(anatListFile):
        raise FileNotFoundError(anatListFile)
    
    anatInputMap = list()

    if anatListFile is not None:
        FID = open(anatListFile, 'r')
        anatList = [x.rstrip() for x in FID.readlines()]
        FID.close()
        
        for curAnatFile in anatList:
            if not os.path.isfile(curAnatFile):
                anatInputMap.append(None)
            else:
                g = nibabel.load(curAnatFile)
                anatInputMap.append(g.agg_data())
                del g
        anatInputMap = numpy.vstack(anatInputMap)

    if os.path.isfile(anatMaskFile):
        anatMask = numpy.loadtxt(anatMaskFile)
        anatMask = anatMask != 0
    
    if os.path.isfile(statDesignFile):
        statDesignMatrix = numpy.loadtxt(statDesignFile)

    if os.path.isfile(eigFile):
        geomEig = scipy.io.loadmat(eigFile)
        geomEig = numpy.float32(geomEig['eig'])
    
    if os.path.isfile(massFile):
        # geomMass = numpy.loadtxt(massFile, delimiter=',')
        geomMass = scipy.io.loadmat(massFile)
        geomMass = numpy.float32(geomMass['mass'].toarray())
    # end of mbm_read_inputs
    
    # mbm_check_read_inputs
    # type and dimension checking
    
    assert statDesignMatrix.shape[0] == anatInputMap.shape[0], "Error. Numbers of subjects in the design matrix and input maps are different."

    if statTest.lower() == "one sample":
        assert statDesignMatrix.shape[1] == 1, "Design matrix for one sample t-test must have one column."
    elif statTest.lower() == "two sample":
        assert statDesignMatrix.shape[1] == 2, "The design matrix for two sample t-test must have two columns."
    elif statTest.lower() == "one way anova":
        assert statDesignMatrix.shape[1] > 1, "The design matrix for one way ANOVA must have at least two columns (two groups)."
        for z in range(1, statDesignMatrix.shape[1]):
            assert numpy.count_nonzero(statDesignMatrix[:, z-1] == 1) == numpy.count_nonzero(statDesignMatrix[:, z] == 1), "Numbers of subjects in each group are different."
    
    assert anatMask.size == anatInputMap.shape[1], "Mask is different from map size"
    assert geomEig.shape[0] == anatMask.size, "'Error. Eigenmodes should be in columns with length compatible with that of the mask"
    
    # dimensions numSubjects
    # anatInputMap should be 2D numSubjects x numVertices
    # mask should be 1D, numVertices
    # statDesignMatrix should be 2D, numSubjects x numGroups
    # geomEig should be 2D, numVertices x numEigs
    # geomMass should be 2D, numVertices x numVertices
    # end mbm_check_read_inputs

    # mask out 
    # inputMap = inputMap(:, MBM.maps.mask == 1);
    
    anatInputMap = numpy.compress(anatMask, anatInputMap, axis=1)
    
    #MBM.eig.eig = MBM.eig.eig(MBM.maps.mask == 1, 1:MBM.eig.nEigenmode);
    geomEig = numpy.take(numpy.compress(anatMask, geomEig, axis=0), numpy.arange(nEigenmodes), axis=1)
    
    #MBM.eig.mass = MBM.eig.mass(MBM.maps.mask == 1, MBM.maps.mask == 1);
    geomMass = numpy.compress(anatMask, geomMass, axis=0)
    geomMass = numpy.compress(anatMask, geomMass, axis=1)

    observedStatMap = mbm_stat_map(anatInputMap, statDesignMatrix, statTest)

    # mbm_perm_test_map
    
    statMapNull = list()#numpy.zeros((statNPer, anatInputMap.shape[2]), dtype=numpy.float32)

    numSubjects = anatInputMap.shape[0]
    numVertices = anatInputMap.shape[1]
    
    # test with trangs code
    iNull = scipy.io.matlab.loadmat('../MBM/iNull.mat')
    iNull = numpy.array(iNull['iNull']) - 1
    for permutationIDX in range(statNPer):
            
        if statTest.lower() == "one sample":

            iSwap = (numpy.float32(numpy.random.uniform(size=(numSubjects, 1)) > 0.5) - 0.5) * 2
            anatInputMapNull = anatInputMap * iSwap

            # the below code has a bug when numpy.sign == 0
            #anatInputMapNull = anatInputMap * numpy.sign(numpy.random.uniform(low=-0.5, high=0.5, size=(numSubjects, 1)))
            statMapNull.append(mbm_stat_map(anatInputMapNull, statDesignMatrix, statTest))
        else:
            #iNull = numpy.argsort(numpy.random.uniform(size=numSubjects))

            statNullDesignMatrix = numpy.take(statDesignMatrix, iNull[:, permutationIDX], axis=0)
            statMapNull.append(mbm_stat_map(anatInputMap, statNullDesignMatrix, statTest))
    statMapNull = numpy.vstack(statMapNull)

    permPMap = numpy.zeros(numVertices, dtype=numpy.float32)
    permRevMap = numpy.zeros(numVertices, dtype=numpy.bool_)

    for vertexIDX in range(numVertices):
        permPMap[vertexIDX], permRevMap[vertexIDX] = mbm_estimate_p_val_tail(statMapNull[:, vertexIDX], observedStatMap[vertexIDX], statPThr)
        
    if statFDR:
        #h, crit_p, adj_ci_cvrg, permPMap = fdr_bh(permPMap, statPThr, 'pdep')
        permPMap = fdr_bh(permPMap, statPThr, 'pdep')[3]
        # del h
        # del crit_p
        # del adj_ci_cvrg

    # mbm_perm_test_map end

    observedStatMapThresh = numpy.sign(observedStatMap)
    observedStatMapThresh[permPMap > statPThr] = 0

    # normalize the eigenmodes
    for z in range(geomEig.shape[1]):
        N = numpy.sqrt(numpy.sum(geomEig[:, z] * geomEig[:, z]))

        if N > 0:
            geomEig[:, z] = geomEig[:, z] / N
        else:
            raise ValueError("Norm zero")
        del N
    
    
    #calc_eigendecomposition    
    eigBeta = neuromodes.basis.decompose(observedStatMap, geomEig, 'project', geomMass, False)

    # mbm_perm_test_beta

    betaNull = neuromodes.basis.decompose(statMapNull.T, geomEig, 'project', geomMass, False)
    betaNull = betaNull.T

    eigPBeta = numpy.zeros(nEigenmodes, dtype=numpy.float32)
    eigPRevBeta = numpy.zeros(nEigenmodes, dtype=numpy.bool_)

    for z in range(nEigenmodes):
        eigPBeta[z], eigPRevBeta[z] = mbm_estimate_p_val_tail(betaNull[:, z], eigBeta[z], statPThr)
    
    if statFDR:
        #h, crit_p, adj_ci_cvrg, permPMap = fdr_bh(eigPBeta, statPThr, 'pdep')
        eigPBeta = fdr_bh(eigPBeta, statPThr, 'pdep')[3]
        # del h
        # del crit_p
        # del adj_ci_cvrg

    # mbm_perm_test_beta end

    significantEigBeta = numpy.array(eigBeta).ravel()
    significantEigBeta[eigPBeta > statPThr] = 0
    
    betaOrderIDX = numpy.argsort(numpy.abs(significantEigBeta).ravel())
    betaOrderIDX = numpy.flip(betaOrderIDX)
    
    betaOrderIDX[numpy.count_nonzero(significantEigBeta != 0):] = -1
    reconMap = numpy.dot(numpy.atleast_2d(significantEigBeta.ravel()), geomEig.T)

    D = dict()
    D['geomMass'] = geomMass
    D['geomEig'] = geomEig
    D['statDesignMatrix'] = statDesignMatrix
    D['anatMask'] = anatMask
    D['anatInputMap'] = anatInputMap
    D['statMapNull'] = statMapNull
    D['permPMap'] = permPMap
    D['permRevMap'] = permRevMap
    D['eigBeta'] = eigBeta
    D['eigPBeta'] = eigPBeta
    D['eigPRevBeta'] = eigPRevBeta
    D['betaNull'] = betaNull
    D['significantEigBeta'] = significantEigBeta
    D['reconMap'] = reconMap
    D['observedStatMap'] = observedStatMap
    D['observedStatMapThresh'] = observedStatMapThresh
    D['betaOrderIDX'] = betaOrderIDX
    return D


if __name__ == "__main__":

    # mbm_demo_sim

    f = mbm(
        anatListFile=os.path.join('..', 'MBM', 'data', 'demo_sim', 'inputMaps_full_path.txt'),
        anatMaskFile=os.path.join('..', 'MBM', 'data', 'demo_sim', 'mask_S1200.L.midthickness_MSMAll.32k_fs_LR.txt'),
        statTest='two sample',
        statDesignFile=os.path.join('..', 'MBM', 'data', 'demo_sim', 'G_onewayANOVA_twosample.txt'),
        statNPer=10,
        statPThr=0.1,
        statThresh=0.05,
        statFDR=True,
        eigFile=os.path.join('..', 'MBM', 'data', 'demo_sim', 'fsLR_32k_midthickness-lh_emode_150.mat'),
        massFile=os.path.join('..', 'MBM', 'data', 'demo_sim', 'fsLR_32k_midthickness-lh_mass_150.mat'),
        nEigenmodes=150
    )
    #f['geomMass'] = numpy.float32(f['geomMass'])
    scipy.io.matlab.savemat('neuromodes_mbm.mat', f)