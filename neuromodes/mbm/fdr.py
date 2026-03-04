import numpy

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
