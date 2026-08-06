import numpy as np
import scipy.stats as sps
import pytest

from neuromodes.glmw import ttestw, ftestw

class TestTestw:
    @staticmethod
    def _compare_ttest_results(act, exp, rtol=1e-9, atol=1e-9):
        np.testing.assert_allclose(act.statistic, exp.statistic, rtol=rtol, atol=atol)
        np.testing.assert_allclose(act.pvalue, exp.pvalue, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("N", [10, 50, 100])
    @pytest.mark.parametrize("alternative", ['two-sided', 'greater', 'less'])
    def test_ttest_1samp_mass_identity(self, N, alternative):
        M = 10

        # Design
        X = np.ones((N, 1))

        # Data
        Y = np.random.normal(size=(N, M)) + np.arange(M)

        # Expected
        exp = sps.ttest_1samp(
            Y, 
            popmean=0, 
            axis=0, 
            alternative=alternative
        )

        # Custom
        act = ttestw(
            X,
            Y, 
            contrast=np.array([[1]]), 
            mass=np.eye(N),
            alternative=alternative
        )

        self._compare_ttest_results(act, exp)

    @pytest.mark.parametrize("N", [[10,10], [20, 30], [50, 100]])
    @pytest.mark.parametrize("alternative", ['two-sided', 'greater', 'less'])
    def test_ttest_ind_mass_identity(self, N, alternative):
        """Compares glmw_test against scipy.stats.ttest_ind (equal variance)."""
        n1, n2 = N
        M = 10
        
        # Design
        X = np.zeros((n1 + n2, 2))
        X[:n1, 0] = 1
        X[n1:, 1] = 1

        # Data
        Y = X @ (np.random.randn(2, M) + np.arange(M)) + np.random.randn(n1 + n2, M)

        # Scipy built-in independent 2-sample t-test (Group 2 vs Group 1)
        exp = sps.ttest_ind(
            Y[n1:], 
            Y[:n1], 
            axis=0, 
            equal_var=True, 
            alternative=alternative
        )

        # Custom
        act = ttestw(
            X, 
            Y, 
            contrast=np.array([[-1, 1]]),   # Contrast for Group 2 > Group 1
            mass=np.eye(n1 + n2),           # mass
            alternative=alternative
        )

        self._compare_ttest_results(act, exp)

    @pytest.mark.parametrize("N", [10, 20, 50])
    @pytest.mark.parametrize("alternative", ['two-sided', 'greater', 'less'])
    def test_ttest_rel_mass_identity(self, N, alternative):
        """Compares ttestw against scipy.stats.ttest_rel."""
        M = 10

        # Design
        # subject intercepts + condition effect
        subject = np.vstack([
            np.eye(N),
            np.eye(N),
        ])

        condition = np.vstack([
            np.zeros((N, 1)),
            np.ones((N, 1)),
        ])

        X = np.hstack([
            subject,
            condition,
        ]) 

        # Data
        Y1 = np.random.randn(N, M)
        Y2 = Y1 + np.arange(M) + np.random.randn(N, M)
        Y = np.vstack([Y1, Y2])

        # Scipy paired t-test
        # Tests condition 1 - condition 2
        exp = sps.ttest_rel(
            Y1,
            Y2,
            axis=0,
            alternative=alternative,
        )

        # GLM contrast:
        # beta_condition = condition 2 - condition 1 to match scipy's Y1 - Y2 convention
        contrast = np.concatenate((np.zeros((1, N)), [[-1]]), axis=1)

        act = ttestw(
            X,
            Y,
            contrast=contrast,
            mass=np.eye(2 * N),
            alternative=alternative,
        )

        self._compare_ttest_results(act, exp)

    @pytest.mark.parametrize("alternative", ['two-sided', 'greater', 'less'])
    def test_ttest_1samp_mass_uniform(self, alternative):
        '''Results should be the same as long as mass is uniform (doesn't have to be 1)'''
        N = 100
        M = 5
        X = np.ones((N,1))
        Y = np.random.normal(size=(N, M)) + np.arange(M)
        exp = sps.ttest_1samp(Y, popmean=0, axis=0, alternative=alternative)
        act = ttestw(X, Y, contrast=np.array([[1]]), 
                    mass=np.eye(N)*np.abs(np.random.random()),
                    alternative=alternative)
        self._compare_ttest_results(act, exp)

    @pytest.mark.parametrize("N", [10, 20, 50])
    @pytest.mark.parametrize("alternative", ['two-sided', 'greater', 'less'])
    def test_ttest_1samp_mass_diagonal(self, N, alternative):
        M = 10
        X = np.ones((N, 1))
        Y = np.random.normal(size=(N, M)) + np.arange(M)
        mass = np.diag(np.random.uniform(0.1, 10, size=N))

        t1 = ttestw(X, 
                    Y, 
                    contrast=np.array([[1]]), 
                    mass=mass, 
                    alternative=alternative)
        
        t2 = ttestw(np.sqrt(mass) @ X, 
                    np.sqrt(mass) @ Y, 
                    contrast=np.array([[1]]), 
                    mass=np.eye(N), 
                    alternative=alternative)

        self._compare_ttest_results(t1, t2)

    @pytest.mark.parametrize("N", [10, 20, 50])
    @pytest.mark.parametrize("alternative", ['two-sided', 'greater', 'less'])
    def test_ttest_1samp_mass_consistent(self, N, alternative):
        M = 10
        X = np.ones((N, 1))
        Y = np.random.normal(size=(N, M)) + np.arange(M)
        # ensure positive semi-definite mass with known square root
        sqrtmass = np.random.random(size=(N, N)) 
        mass = sqrtmass.T @ sqrtmass

        t1 = ttestw(X, 
                    Y, 
                    contrast=np.array([[1]]), 
                    mass=mass, 
                    alternative=alternative)
        
        t2 = ttestw(sqrtmass @ X, 
                    sqrtmass @ Y, 
                    contrast=np.array([[1]]), 
                    mass=np.eye(N), 
                    alternative=alternative)

        self._compare_ttest_results(t1, t2)

    @pytest.mark.parametrize("C", [1, 2, 50])
    @pytest.mark.parametrize("alternative", ['two-sided', 'greater', 'less'])
    def test_ttest_1samp_contrast_scale(self, C, alternative):
        N = 100
        M = 10
        X = np.ones((N, 1))
        Y = np.random.normal(size=(N, M)) + np.arange(M)

        exp = sps.ttest_1samp(Y, popmean=0, axis=0, alternative=alternative)
        act = ttestw(X, Y, contrast=np.array([[C]]), mass=np.eye(N), alternative=alternative)

        self._compare_ttest_results(act, exp)

    # TODO consider adding more tests for ttestw with non-identity mass matrices

class TestFtestw:
    @staticmethod
    def _compare_ftest_results(act, exp, rtol=1e-9, atol=1e-9):
        np.testing.assert_allclose(act[0], exp[0], rtol=rtol, atol=atol)
        np.testing.assert_allclose(act[1], exp[1], rtol=rtol, atol=atol)

    @pytest.mark.parametrize("N", [[10,10], [10,10,20], [10,20,30,40]])
    def test_f_oneway_mass_identity(self, N):
        """Compares glmw_test against scipy.stats.f_oneway across different groups."""
        M = 10
        k = len(N)
        s = sum(N)

        # Design and data 
        X = np.zeros((s, k))
        Y = []
        for i, n in enumerate(N):
            X[sum(N[:i]):sum(N[:i+1]), i] = 1
            Y.append(np.random.normal(size=(n, M)) + np.arange(M))

        # Standard omnibus ANOVA contrast
        contrast = np.hstack([ 
            np.ones((k - 1,1)), 
            -np.eye(k - 1) 
            ])
        
        # Expected
        exp = sps.f_oneway(*Y, axis=0)

        # Custom
        act = ftestw(X, np.vstack(Y), contrast, mass=np.eye(s))

        self._compare_ftest_results(act, exp)

    @pytest.mark.parametrize("N", [[10, 10], [10, 10, 20], [10, 20, 30, 40]])
    def test_f_oneway_mass_consistent(self, N):
        """Mass matrix should be equivalent to whitening the design and data."""
        M = 10
        k = len(N)
        s = sum(N)

        # Design and data
        X = np.zeros((s, k))
        Y = []
        for i, n in enumerate(N):
            X[sum(N[:i]):sum(N[:i+1]), i] = 1
            Y.append(np.random.normal(size=(n, M)) + np.arange(M))

        Y = np.vstack(Y)

        # Standard omnibus ANOVA contrast
        contrast = np.hstack([
            np.ones((k - 1, 1)),
            -np.eye(k - 1),
        ])

        # Positive-definite mass matrix with known square root
        sqrtmass = np.random.random((s, s))
        mass = sqrtmass.T @ sqrtmass

        f1 = ftestw(
            X,
            Y,
            contrast,
            mass,
        )

        f2 = ftestw(
            sqrtmass @ X,
            sqrtmass @ Y,
            contrast,
            np.eye(s),
        )

        self._compare_ftest_results(f1, f2)
