
import pytest
import numpy as np
import scipy.stats as sps

from neuromodes.glmw import glmw_estimate, glmw_test

import numpy as np
import pytest


# =====================================================================
# 1. ONE-SAMPLE SETUP
# =====================================================================
def test_recover_one_sample():
    """Tests if glmw_estimate recovers the grand mean (intercept)."""
    n_samples = 40
    n_features = 3
    
    # Design Matrix: Just a column of ones (Intercept)
    X = np.ones((n_samples, 1))
    
    # True Beta: Shape (n_predictors, n_features) -> (1, 3)
    true_beta = np.random.rand(1, n_features) 
    
    # Generate noiseless response variables: Y = X * Beta
    Y = X.dot(true_beta)
    I_mass = np.eye(n_samples)

    beta_recovered, residuals = glmw_estimate(I_mass, Y, X)

    # Beta should be perfectly recovered, residuals should be 0
    np.testing.assert_allclose(beta_recovered, true_beta, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(residuals, 0.0, rtol=1e-8, atol=1e-8)


# =====================================================================
# 2. TWO-SAMPLE SETUP
# =====================================================================
def test_recover_two_sample():
    """Tests if glmw_estimate recovers group means for a two-sample layout."""
    n1, n2 = 25, 25
    n_samples = n1 + n2
    n_features = 2

    # Design Matrix: Indicator columns for Group 1 and Group 2
    X = np.zeros((n_samples, 2))
    X[:n1, 0] = 1.0
    X[n1:, 1] = 1.0

    # True Betas: Rows are groups, columns are features
    true_beta = np.random.rand(2, n_features)

    Y = X.dot(true_beta)
    I_mass = np.eye(n_samples)

    beta_recovered, residuals = glmw_estimate(I_mass, Y, X)

    np.testing.assert_allclose(beta_recovered, true_beta, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(residuals, 0.0, rtol=1e-8, atol=1e-8)


# =====================================================================
# 3. ONE-WAY ANOVA SETUP
# =====================================================================
def test_recover_oneway_anova():
    """Tests parameter recovery for a 3-group One-Way ANOVA design."""
    n_per_group = 20
    n_groups = 3
    n_samples = n_per_group * n_groups
    n_features = 2

    # Design Matrix: Cell means coding for 3 groups
    X = np.zeros((n_samples, n_groups))
    for i in range(n_groups):
        X[i*n_per_group:(i+1)*n_per_group, i] = 1.0

    # True Betas: (3 predictors, 2 features)
    true_beta = np.random.rand(n_groups, n_features)

    Y = X.dot(true_beta)
    I_mass = np.eye(n_samples)

    beta_recovered, residuals = glmw_estimate(I_mass, Y, X)

    np.testing.assert_allclose(beta_recovered, true_beta, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(residuals, 0.0, rtol=1e-8, atol=1e-8)


# =====================================================================
# 4. ANCOVA SETUP
# =====================================================================
def test_recover_ancova():
    """Tests parameter recovery for an ANCOVA design.
    
    Model: Intercept, Group_Offset (Categorical), and a Continuous Covariate.
    """
    n_per_group = 30
    n_samples = n_per_group * 2
    n_features = 2

    # Continuous covariate (e.g., age centered around 0)
    np.random.seed(0)
    covariate = np.linspace(-10, 10, n_samples)

    # Design Matrix: [Intercept, Group2_Indicator, Covariate]
    X = np.zeros((n_samples, 3))
    X[:, 0] = 1.0               # Intercept (Group 1 baseline)
    X[n_per_group:, 1] = 1.0    # Group 2 offset shift
    X[:, 2] = covariate         # Covariate slope effect

    # True Betas: (3 predictors, 2 features)
    true_beta = np.random.rand(3, n_features)

    Y = X.dot(true_beta)
    I_mass = np.eye(n_samples)

    beta_recovered, residuals = glmw_estimate(I_mass, Y, X)

    np.testing.assert_allclose(beta_recovered, true_beta, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(residuals, 0.0, rtol=1e-8, atol=1e-8)


# =====================================================================
# 1. ONE-SAMPLE T-TEST
# =====================================================================
def test_one_sample_ttest():
    """Compares glmw_test against scipy.stats.ttest_1samp."""
    n_samples = 50
    n_targets = 3

    # Pure base shifted by a population mean + small controlled gradient
    pop_mean = 5.0
    Y = np.ones((n_samples, n_targets)) * pop_mean
    Y += np.linspace(-0.5, 0.5, n_samples)[:, None]

    # Model: Y = intercept
    X = np.ones((n_samples, 1))
    I_mass = np.eye(n_samples)

    # Test null hypothesis: mean = 0
    contrast = np.array([[1]])

    t_custom, p_custom = glmw_test(Y, X, I_mass, contrast)

    # Scipy built-in 1-sample t-test
    res_sp = sps.ttest_1samp(Y, popmean=0, axis=0, alternative='greater')  # Right-sided test

    # Convert scipy's two-sided p-value to right-sided (1 - cdf) to match glmw_test
    # p_expected_right = res_sp.pvalue / 2
    # p_expected_right = np.where(
    #     res_sp.statistic > 0, p_expected_right, 1 - p_expected_right
    # )

    np.testing.assert_allclose(t_custom, res_sp.statistic, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(p_custom, res_sp.pvalue, rtol=1e-8, atol=1e-8)


# =====================================================================
# 2. TWO-SAMPLE T-TEST
# =====================================================================
def test_two_sample_ttest():
    """Compares glmw_test against scipy.stats.ttest_ind (equal variance)."""
    n1, n2 = 30, 30
    n_targets = 2

    # Group separation + small controlled variation
    group1 = np.ones((n1, n_targets)) * 10.0 + np.linspace(-0.1, 0.1, n1)[:, None]
    group2 = np.ones((n2, n_targets)) * 12.0 + np.linspace(-0.1, 0.1, n2)[:, None]
    Y = np.vstack([group1, group2])

    # Design matrix: Col 0 = Group 1, Col 1 = Group 2
    X = np.zeros((n1 + n2, 2))
    X[:n1, 0] = 1
    X[n1:, 1] = 1

    I_mass = np.eye(n1 + n2)

    # Contrast for Group 2 > Group 1: [-1, 1]
    contrast = np.array([[-1, 1]])

    t_custom, p_custom = glmw_test(Y, X, I_mass, contrast)

    # Scipy built-in independent 2-sample t-test (Group 2 vs Group 1)
    res_sp = sps.ttest_ind(group2, group1, axis=0, equal_var=True, alternative='greater')  # Right-sided test

    np.testing.assert_allclose(t_custom, res_sp.statistic, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(p_custom, res_sp.pvalue, rtol=1e-8, atol=1e-8)


# =====================================================================
# 3. ONE-WAY ANOVA
# =====================================================================
def test_oneway_anova():
    """Compares glmw_test against scipy.stats.f_oneway across 3 groups."""
    n_per_group = 20
    n_targets = 1

    # 3 distinct groups
    g1 = np.ones((n_per_group, n_targets)) * 5.0 + np.linspace(-0.2, 0.2, n_per_group)[:, None]
    g2 = np.ones((n_per_group, n_targets)) * 7.0 + np.linspace(-0.2, 0.2, n_per_group)[:, None]
    g3 = np.ones((n_per_group, n_targets)) * 9.0 + np.linspace(-0.2, 0.2, n_per_group)[:, None]
    Y = np.vstack([g1, g2, g3])

    # Cell means design matrix
    X = np.zeros((n_per_group * 3, 3))
    X[:n_per_group, 0] = 1
    X[n_per_group : 2 * n_per_group, 1] = 1
    X[2 * n_per_group :, 2] = 1

    I_mass = np.eye(n_per_group * 3)

    # Omnibus contrast: Group 1 = Group 2 AND Group 2 = Group 3
    contrast = np.array([[1, -1, 0], [0, 1, -1]])

    f_custom, p_custom = glmw_test(Y, X, I_mass, contrast)

    # Scipy built-in standard F-oneway ANOVA
    f_sp, p_sp = sps.f_oneway(g1, g2, g3, axis=0)

    np.testing.assert_allclose(f_custom, f_sp, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(p_custom, p_sp, rtol=1e-8, atol=1e-8)

