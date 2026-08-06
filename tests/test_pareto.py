import numpy as np
from scipy.stats import genpareto
import pytest
from neuromodes.palm import palm_pareto

@pytest.mark.parametrize("seed", [42, np.random.randint(0, 10000)])
@pytest.mark.parametrize("u_true", [-1, 0, 5, 10.0]) # infinite support (location)
@pytest.mark.parametrize("k_true", [-0.2, -0.1, 0, 0.1, 0.2]) # must be above -0.5, typically up to around 0.2
@pytest.mark.parametrize("a_true", [1, 5, 10]) # positive
def test_palm_pareto_accuracy(seed, u_true, k_true, a_true):    
    # 1. Generate samples from a known GPD
    np.random.seed(seed)
    c_true = -k_true  # SciPy parameterization (c = -k)
    
    n_samples = 1_000_000
    Gdist = genpareto.rvs(c=c_true, loc=u_true, scale=a_true, size=n_samples)
    
    # 2. Setup an outlier test vector G
    G = np.array([np.percentile(Gdist, 99)]) # choose an outlier so the tail estimation is forced
    
    # 3. Execute palm_pareto
    P, apar, kpar, upar = palm_pareto(
        G=G,
        Gdist=Gdist,
        rev=False,
        Pthr=0.05,
        G1out=False
    )
    
    # 4. Determine theoretical expected upar:
    # this will differ from the above value as the calculation is done using <=25% of the values
    Q_USED = np.mean(Gdist <= upar)
    expected_upar_theoretical = genpareto.ppf(Q_USED, c=c_true, loc=u_true, scale=a_true)
    # The accuracy of a depends on the accuracy of u
    expected_apar_theoretical = a_true - k_true * (upar - u_true)

    # Location parameter (upar) converges tightly
    np.testing.assert_allclose(
        upar, expected_upar_theoretical, rtol=0.01, atol=0.01
    )

    # Shape parameter (kpar)
    # Need to have some tolerance at k=0 (error generally very small but can't use rtol)
    # and and large |k| where the error is expected to be larger
    np.testing.assert_allclose(
        kpar, k_true, atol=max(0.01, 0.15 * abs(k_true))
    )

    # Scale parameter (apar) — needs ~5% tolerance for heavy tails (k = -0.20)
    np.testing.assert_allclose(
        apar, expected_apar_theoretical, rtol=max(0.02, 0.25 * abs(k_true))
    )
