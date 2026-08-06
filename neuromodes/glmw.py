import numpy as np
import scipy.stats as sps


# TODO rename n_vertices and n_subjects to ?n_samples and ?n_features for generality
def glmw_estimate(
        massMatrix, # (n_vertices, n_vertices)
        responseVariables, # (n_vertices, n_data)
        designMatrix # (n_vertices, n_predictors)
):
    # Prelims TODO change to eigendata
    X = np.asarray(designMatrix)
    Y = np.asarray(responseVariables)
    n_vertices = X.shape[0]
    if Y.shape[0] != n_vertices or massMatrix.shape[0] != n_vertices:
        raise ValueError("Dimension mismatch across inputs along the vertex axis.")

    # Since massMatrix is sparse, compute (M * X).T to keep it clean and efficient
    XtM = (massMatrix.dot(X)).T # Shape: (n_predictors, n_vertices)

    # Solve for beta: beta = (X^T * M * X)^(-1) * X^T * M * Y
    beta = np.linalg.solve(XtM.dot(X), XtM.dot(Y))
    residuals = Y - X.dot(beta)
    
    return beta, residuals


# TODO consider splitting into ftestw and ttestw ?ttest1w and ttest2w
# TODO consider allowing tail for t test (alternative='greater','less','two-sided')
def glmw_test(
        responseVariables, # (n_vertices, n_data)
        designMatrix, # (n_vertices, n_predictors)
        massMatrix, # (n_vertices, n_vertices)
        contrastMatrix, # (n_contrasts, n_predictors)
        alternative = 'two-sided', # 'two-sided', 'greater', 'less' # TODO validate here
):
    # 1. Fit GLM
    beta, residuals = glmw_estimate(massMatrix, responseVariables, designMatrix)
    
    X = np.asarray(designMatrix)
    Y_res = np.asarray(residuals)
    C = np.atleast_2d(contrastMatrix)
    
    # 2. Shared Data Metrics
    weighted_ssr = np.sum(Y_res * massMatrix.dot(Y_res), axis=0) # TODO replace with ssqw
    dof = X.shape[0] - np.linalg.matrix_rank(X)
    sigma_sq = weighted_ssr / dof
    
    # 3. Unified Hypothesis Transformation
    D = C.dot(beta)  # Shape: (n_contrasts, n_data)

    A = (massMatrix.dot(X)).T.dot(X)
    G = C.dot(np.linalg.solve(A, C.T))  # Shape: (n_contrasts, n_contrasts)
    V = np.linalg.solve(G, D) # A and G not used after this

    Q = np.maximum(np.sum(D * V, axis=0), 0) / sigma_sq # V not used after this

    # 4. T and F stats
    if C.shape[0] == 1: # one and two sample t-tests
        # Recover directional t-statistic: t = sign(D) * sqrt(Q)
        t_stat = np.sign(D.flatten()) * np.sqrt(Q)
        
        p_vals = sps.t.cdf(t_stat, df=dof) # left sided tailed test
        if alternative == 'two-sided':
            p_vals = 2 * np.minimum(p_vals, 1 - p_vals)
        elif alternative == 'greater':
            p_vals = 1 - p_vals
        elif alternative == 'less':
            pass
        else:
            raise ValueError(f"Unsupported alternative hypothesis: {alternative}")
            
        return t_stat, p_vals
        
    else: # one way anova and ancova
        # Compute omnibus F-statistic: F = Q / rank(C)
        rank_C = np.linalg.matrix_rank(C)
        f_stat = Q / rank_C
        p_vals = 1 - sps.f.cdf(f_stat, dfn=rank_C, dfd=dof)
        
        return f_stat, p_vals

