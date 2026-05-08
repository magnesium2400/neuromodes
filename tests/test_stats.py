import numpy as np
import pytest
from scipy.spatial.distance import cdist, pdist
from scipy.stats import zscore
from neuromodes.stats import (gramw, dotw, ssqw, lstsqw, solvew, cdistw, pdistw, meanw, demeanw,
                              varw, stdw, zscorew, covw, vecnormw, correlationw)

@pytest.fixture(scope='module')
def random_data():
    """Generate random data for testing."""
    rng = np.random.default_rng(0)
    n_verts = 10
    n_maps = 3

    X = rng.standard_normal(size=(n_verts, n_maps))
    Y = rng.standard_normal(size=(n_verts, n_maps))

    mass = np.eye(n_verts) # identity mass for testing
    return X, Y, mass

def test_gramw_eye(random_data):
    """Test that Gram matrix with identity mass is the same as unweighted Gram matrix."""
    X, Y, mass = random_data
    gram_w = gramw(X, Y, mass=mass)
    gram_u = X.T @ Y
    assert np.allclose(gram_w, gram_u), "Gram matrix with identity mass should be the same as " \
                                        "unweighted Gram matrix."

def test_dotw_eye(random_data):
    """Test that dot product with identity mass is the same as unweighted dot product."""
    X, Y, mass = random_data
    dot_w = dotw(X, Y, mass=mass)
    dot_u = np.einsum('ij,ij->j', X, Y)
    assert np.allclose(dot_w, dot_u), "Dot product with identity mass should be the same as " \
                                     "unweighted dot product."
    
def test_ssqw_eye(random_data):
    """Test that sum of squares with identity mass is the same as unweighted sum of squares."""
    X, _, mass = random_data
    ssq_w = ssqw(X, mass)
    ssq_u = np.sum(X**2, axis=0)
    assert np.allclose(ssq_w, ssq_u), "Sum of squares with identity mass should be the same as " \
                                      "unweighted sum of squares."
    
def test_meanw_eye(random_data):
    """Test that mean with identity mass is the same as unweighted mean."""
    X, _, mass = random_data
    mean_w = meanw(X, mass=mass)
    mean_u = np.mean(X, axis=0)
    assert np.allclose(mean_w, mean_u), "Mean with identity mass should be the same as " \
                                        "unweighted mean."
    
def test_varw_eye(random_data):
    """Test that variance with identity mass is the same as unweighted variance."""
    X, _, mass = random_data
    var_w = varw(X, mass=mass)
    var_u = np.var(X, axis=0, ddof=0)
    assert np.allclose(var_w, var_u), "Variance with identity mass should be the same as " \
                                      "unweighted variance."
    
def test_stdw_eye(random_data):
    """Test that standard deviation with identity mass is the same as unweighted standard deviation."""
    X, _, mass = random_data
    std_w = stdw(X, mass=mass)
    std_u = np.std(X, axis=0)
    assert np.allclose(std_w, std_u), "Standard deviation with identity mass should be the same " \
                                      "as unweighted standard deviation."
    
def test_zscorew_eye(random_data):
    """Test that z-score with identity mass is the same as unweighted z-score."""
    X, _, mass = random_data
    zscore_w = zscorew(X, mass=mass)
    zscore_u = zscore(X)
    assert np.allclose(zscore_w, zscore_u), "Z-score with identity mass should be the same as " \
                                            "unweighted z-score."

def test_correlationw_eye(random_data):
    """Test that correlation with identity mass is the same as unweighted correlation."""
    X, Y, mass = random_data
    corr_w = correlationw(X, Y, mass=mass)
    corr_u = np.corrcoef(X, Y, rowvar=False)[:X.shape[1], X.shape[1]:]
    assert np.allclose(corr_w, corr_u), "Correlation with identity mass should be the same as " \
                                        "unweighted correlation."

def test_momentw_eye(random_data):
    # TODO: decide whether/how to test this
    pass

def test_covw_eye(random_data):
    """Test that covariance with identity mass is the same as unweighted covariance."""
    X, Y, mass = random_data
    cov_w = covw(X, Y, mass=mass)
    cov_u = np.cov(X, Y, rowvar=False)[:X.shape[1], X.shape[1]:]
    assert np.allclose(cov_w, cov_u), "Covariance with identity mass should be the same as " \
                                      "unweighted covariance."

def test_covw_unbiased_eye(random_data):
    """Test that unbiased covariance with identity mass matches NumPy's default covariance."""
    X, Y, mass = random_data
    cov_w = covw(X, Y, mass=mass, bias=False)
    cov_u = np.cov(X, Y, bias=False, rowvar=False)[:X.shape[1], X.shape[1]:]
    assert np.allclose(cov_w, cov_u), "Unbiased covariance with identity mass should match " \
                                      "NumPy's default covariance."

def test_vecnormw_eye(random_data):
    """Test that vector norm with identity mass is the same as unweighted vector norm."""
    X, _, mass = random_data
    norm_w = vecnormw(X, mass=mass)
    norm_u = np.linalg.norm(X, axis=0)
    assert np.allclose(norm_w, norm_u), "Vector norm with identity mass should be the same as " \
                                        "unweighted vector norm."
    
def test_demeanw_eye(random_data):
    """Test that demean with identity mass is the same as unweighted demean."""
    X, _, mass = random_data
    demean_w = demeanw(X, mass=mass)
    demean_u = X - np.mean(X, axis=0)
    assert np.allclose(demean_w, demean_u), "Demean with identity mass should be the same as " \
                                            "unweighted demean."
    
def test_lstsqw_eye(random_data):
    """Test that least squares with identity mass is the same as unweighted least squares."""
    X, Y, mass = random_data
    lstsq_w = lstsqw(X, Y, mass=mass)[0]
    lstsq_u = np.linalg.lstsq(X, Y, rcond=None)[0]
    assert np.allclose(lstsq_w, lstsq_u), "Least squares with identity mass should be the same " \
                                          "as unweighted least squares."\

def test_solvew_eye(random_data):
    """Test that solve with identity mass is the same as unweighted solve."""
    X, Y, mass = random_data

    solve_w = solvew(X, Y, mass=mass)
    solve_u = np.linalg.solve(X.T @ X, X.T @ Y)
    assert np.allclose(solve_w, solve_u), "Solve with identity mass should be the same as " \
                                          "unweighted solve."
    
def test_cdistw_eye(random_data):
    """Test that pairwise distances with identity mass are the same as unweighted pairwise distances."""
    X, Y, mass = random_data
    for metric in ['euclidean', 'sqeuclidean', 'cosine', 'correlation']:
        cdist_w = cdistw(X, Y, mass=mass, metric=metric)
        cdist_u = cdist(X.T, Y.T, metric=metric)
        assert np.allclose(cdist_w, cdist_u), "Pairwise distances with identity mass should be " \
                                              "the same as unweighted pairwise distances for " \
                                              f"metric '{metric}'."
    
def test_pdistw_eye(random_data):
    """Test that pairwise distances with identity mass are the same as unweighted pairwise distances."""
    X, _, mass = random_data

    for metric in ['euclidean', 'sqeuclidean', 'cosine', 'correlation']:
        pdist_w = pdistw(X, mass=mass, metric=metric)
        pdist_u = pdist(X.T, metric=metric)
        assert np.allclose(pdist_w, pdist_u), "Pairwise distances with identity mass should be " \
                                              "the same as unweighted pairwise distances for " \
                                              f"metric '{metric}'."