import numpy as np
from scipy.optimize import minimize
import pytest

from neuromodes.palm import andersondarling, palm_competitive, palm_datapval, _fit_tail, palm_pareto

class Test_andersondarling:
    @staticmethod
    def _a2_to_z(a2_target, n=27):
        """
        Generates a sorted z array yielding the target A^2 statistic
        """
        # Penalty term for A^2 match
        def objective(z):
            a2_calc = -n - np.mean(np.log(z * (1 - z[::-1])) * np.arange(1, 2 * n, 2))
            return (a2_calc - a2_target) ** 2

        # x0 starts with an evenly spaced initial guess in (0, 1) 
        # bounds enforces strict boundaries (0 < z_i < 1)
        # constraints enforces monotonic ordering (z[i-1] <= z[i])
        return minimize(
            objective,
            x0=np.arange(1, n + 1) / (n + 1), 
            method='SLSQP', 
            bounds=[(1e-6, 1 - 1e-6) for _ in range(n)], 
            constraints={'type': 'ineq', 'fun': lambda z: np.diff(z)},
            options={'ftol': 1e-12, 'maxiter': 500}
        ).x

    @staticmethod
    def _check_p(z, k, expected_pval, atol=1e-6):
        p, _ = andersondarling(z, k)
        np.testing.assert_allclose(p, expected_pval, atol=atol)

    def test_simple(self):
        z = self._a2_to_z(2.058)
        self._check_p(z, 0.40, 0.005)

    @pytest.mark.parametrize("target_a2, k, expected_pval", [
        (0.4990, -0.5, 0.250),
        (2.9220,  0.5, 0.001),
        (0.3390, -0.9, 0.50),
    ])
    def test_exact_table_values(self, target_a2, k, expected_pval):
        """Test precise matches for nodes directly on the provided table."""
        z = self._a2_to_z(target_a2)
        self._check_p(z, k, expected_pval)

    @pytest.mark.parametrize("target_a2, expected_pval", [
        (0.468, 0.500), 
        (0.688, 0.250),
        (0.985, 0.100),
        (1.221, 0.050),
        (1.465, 0.025),
        (1.799, 0.010),
        (2.058, 0.005),
        (2.674, 0.001)
    ])
    def test_table_row(self, target_a2, expected_pval):
        """Test interpolation along a single row of the table for k=0."""
        z = self._a2_to_z(target_a2)
        self._check_p(z, 0.400, expected_pval)

    @pytest.mark.parametrize("target_a2, k", [
        (0.641, -0.90),
        (0.685, -0.50),
        (0.741, -0.20),
        (0.766, -0.10),
        (0.796,  0.00),
        (0.831,  0.10),
        (0.873,  0.20),
        (0.924,  0.30),
        (0.985,  0.40),
        (1.061,  0.50)
    ])
    def test_table_column(self, target_a2, k):
        """Test interpolation along a single column of the table for various k values."""
        z = self._a2_to_z(target_a2)
        self._check_p(z, k, 0.100)

    def test_interpolated_midway_row(self):
        """Test 2D interpolation for values between the table's rows and columns."""
        # Midway between rows
        target_a2_row = (0.741 + 0.766) / 2
        z = self._a2_to_z(target_a2_row)
        self._check_p(z, k=-0.15, expected_pval=0.100)

    def test_interpolated_midway_column(self):
        """Test 2D interpolation for values between the table's rows and columns."""
        # Midway between columns
        target_a2_col = (0.569 + 0.796) / 2
        z = self._a2_to_z(target_a2_col)
        self._check_p(z, k=0.0, expected_pval=0.175)

class Test_competitive(): 
    @staticmethod
    def _test_competitive(X, exp_unsrtR, exp_S, exp_srtR, descending=False, ranking='standard'):
        unsrtR, S, srtR = palm_competitive(X, ranking=ranking, descending=descending)
        # Test correctness of results
        assert np.array_equal(S[unsrtR - 1], X)
        assert np.array_equal(np.sort(unsrtR), srtR)
        # Test that expected outputs match actual outputs
        assert np.array_equal(unsrtR, exp_unsrtR)
        assert np.array_equal(S, exp_S)
        assert np.array_equal(srtR, exp_srtR)

    @pytest.mark.parametrize("ranking", ['standard', 'modified'])
    def test_noties_ascending(self, ranking): 
        X = np.random.choice(100, size=100, replace=False)
        self._test_competitive(X, 
                               np.argsort(np.argsort(X)) + 1,
                               np.sort(X),
                               np.arange(X.size) + 1,
                               descending=False,
                               ranking=ranking)

    @pytest.mark.parametrize("ranking", ['standard', 'modified'])
    def test_noties_descending(self, ranking): 
        X = np.random.choice(100, size=100, replace=False)
        self._test_competitive(X,
                               np.argsort(np.argsort(-X)) + 1,  # hack to avoid numpy2.5+
                               -np.sort(-X),
                               np.arange(X.size) + 1,
                               descending=True,
                               ranking=ranking)

    @pytest.mark.parametrize("N", [10, 20, 50])
    @pytest.mark.parametrize("M", [20, 30, 40])
    def test_ties_ascending_standard(self, N, M): 
        X = np.concatenate([np.ones(N), np.zeros(M)])
        self._test_competitive(X,
                               np.concatenate([np.ones(N)*(M+1), np.ones(M)]),
                               np.concatenate([np.zeros(M), np.ones(N)]),
                               np.concatenate([np.ones(M), np.ones(N)*(M+1)]),
                               descending=False,
                               ranking='standard')

    @pytest.mark.parametrize("N", [10, 20, 50])
    @pytest.mark.parametrize("M", [20, 30, 40])
    def test_ties_ascending_modified(self, N, M): 
        X = np.concatenate([np.ones(N), np.zeros(M)])
        self._test_competitive(X,
                               np.concatenate([np.ones(N)*(M+N), np.ones(M)*M]),
                               np.concatenate([np.zeros(M), np.ones(N)]),
                               np.concatenate([np.ones(M)*M, np.ones(N)*(M+N)]),
                               descending=False,
                               ranking='modified')

    @pytest.mark.parametrize("N", [10, 20, 50])
    @pytest.mark.parametrize("M", [20, 30, 40])
    def test_ties_descending_standard(self, M, N): 
        X = np.concatenate([np.zeros(N), np.ones(M)])
        self._test_competitive(X,
                               np.concatenate([np.ones(N)*(M+1), np.ones(M)]),
                               np.concatenate([np.ones(M), np.zeros(N)]),
                               np.concatenate([np.ones(M), np.ones(N)*(M+1)]),
                               descending=True,
                               ranking='standard')

    @pytest.mark.parametrize("N", [10, 20, 50])
    @pytest.mark.parametrize("M", [20, 30, 40])
    def test_ties_descending_modified(self, N, M): 
        X = np.concatenate([np.zeros(N), np.ones(M)])
        self._test_competitive(X,
                               np.concatenate([np.ones(N)*(M+N), np.ones(M)*M]),
                               np.concatenate([np.ones(M), np.zeros(N)]),
                               np.concatenate([np.ones(M)*M, np.ones(N)*(M+N)]),
                               descending=True,
                               ranking='modified')

    @pytest.mark.parametrize("X", [np.array([10, 20, 30, 20])])
    @pytest.mark.parametrize("descending, ranking, exp_unsrtR, exp_S, exp_srtR", [
        (False,     'standard', [1, 2, 4, 2], [10, 20, 20, 30], [1, 2, 2, 4]),
        (False,     'modified', [1, 3, 4, 3], [10, 20, 20, 30], [1, 3, 3, 4]),
        (True,      'standard', [4, 2, 1, 2], [30, 20, 20, 10], [1, 2, 2, 4]),
        (True,      'modified', [4, 3, 1, 3], [30, 20, 20, 10], [1, 3, 3, 4]),
    ])
    def test_ranking_logic_full_1d(self, X, descending, ranking, exp_unsrtR, exp_S, exp_srtR):
        """
        Parametrized test checking unsrtR, S, and srtR.
        
        compare this to matlab output: 
        https://github.com/andersonwinkler/PALM/blob/9086eff76dcacf4afde12c9af5400a1fe0033b9e/palm_competitive.m

        X = [10 20 20 30]';
        for ord = ["ascend", "descend"];
        for mod = [false, true];
        [unsrtR, S, srtR] = palm_competitive(X, ord, mod)
        end
        end
        """

        self._test_competitive(X, 
                               np.array(exp_unsrtR), 
                               np.array(exp_S), 
                               np.array(exp_srtR), 
                               descending=descending, 
                               ranking=ranking)

    @pytest.mark.parametrize("descending", [False, True])
    @pytest.mark.parametrize("ranking", ['standard', 'modified'])
    def test_ranking_logic_full_2d(self, descending, ranking):
        """Check that 2D behaves as expected by testing each column independently."""
        X = np.random.choice(100, size=(10, 5), replace=False)
        # 2D
        unsrtR, S, srtR = palm_competitive(X, ranking=ranking, descending=descending)
        # Compare against 1D results for each column
        for col in range(X.shape[1]):
            self._test_competitive(X[:, col], 
                                   unsrtR[:, col], 
                                   S[:, col], 
                                   srtR[:, col], 
                                   descending=descending, 
                                   ranking=ranking)

    def test_3d_array_ties(self):
        # Shape: (4, 2, 2)
        X = np.array([
            [[10, 100], [1, 2]],
            [[20, 100], [1, 3]],
            [[20, 200], [2, 3]],
            [[30, 200], [2, 4]]
        ])
        
        unsrtR, S, srtR = palm_competitive(X, ranking='standard', descending=False)
        
        # Check individual slices manually
        for i in range(X.shape[1]):
            for j in range(X.shape[2]):
                u, s, sr = palm_competitive(X[:, i, j], ranking='standard', descending=False)
                np.testing.assert_array_equal(unsrtR[:, i, j], u)
                np.testing.assert_array_equal(S[:, i, j], s)
                np.testing.assert_array_equal(srtR[:, i, j], sr)


class Test_datapval:
    @staticmethod
    def _test_datapval(G, Gvals, descending, expected):
        p_vals = palm_datapval(G, Gvals, descending=descending)
        np.testing.assert_allclose(p_vals, expected)

    def test_ascending_basic(self):
        self._test_datapval(
            G=np.array([0.5, 1.0, 2.5, 4.0, 5.0]),
            Gvals=np.array([1.0, 2.0, 3.0, 4.0]),
            descending=False,
            expected=np.array([1.0, 1.0, 0.5, 0.25, 0.0])
        )

    def test_descending_basic(self):
        self._test_datapval(
            G=np.array([0.5, 1.0, 2.5, 4.0, 5.0]),
            Gvals=np.array([1.0, 2.0, 3.0, 4.0]),
            descending=True,
            expected=np.array([0.0, 0.25, 0.5, 1.0, 1.0])
        )

    def test_ascending_ties(self):
        """Test behavior when reference distribution contains duplicate values (ties)."""
        self._test_datapval(
            G=[1.9, 2.0, 2.1], 
            Gvals=np.array([1.0, 2.0, 2.0, 2.0, 3.0]), 
            descending=False, 
            expected=[0.8, 0.8, 0.2]
        )

    def test_descending_ties(self):
        """Test behavior when reference distribution contains duplicate values (ties)."""
        self._test_datapval(
            G=[1.9, 2.0, 2.1], 
            Gvals=np.array([1.0, 2.0, 2.0, 2.0, 3.0]), 
            descending=True, 
            expected=[0.2, 0.8, 0.8]
        )

    def test_ascending_single(self): 
        """Edge case: reference set contains only 1 value."""
        self._test_datapval(
            G=[0.0, 1.0, 2.0], 
            Gvals=np.array([1.0]), 
            descending=False, 
            expected=[1.0, 1.0, 0.0]
        )

    def test_descending_single(self):
        """Edge case: reference set contains only 1 value."""
        self._test_datapval(
            G=[0.0, 1.0, 2.0], 
            Gvals=np.array([1.0]), 
            descending=True, 
            expected=[0.0, 1.0, 1.0]
        )    

    @pytest.mark.parametrize("descending", [False, True])
    def test_output_shape_preservation(self, descending):
        """Ensure input array shape of G is strictly preserved in output."""
        Gvals = np.linspace(0, 10, 100)
        G_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        p_vals = palm_datapval(G_2d, Gvals, descending=descending)
        assert p_vals.shape == G_2d.shape

    @pytest.mark.parametrize("descending", [False, True])
    def test_bounds_clipping(self, descending):
        """Check that output probabilities are always strictly bounded within [0.0, 1.0]."""
        Gvals = np.array([10, 20, 30])
        G = np.array([-np.inf, -1000, 0, 15, 1000, np.inf])
        p_vals = palm_datapval(G, Gvals, descending=descending)
        assert np.all((p_vals >= 0.0) & (p_vals <= 1.0))

class Test_fit_tail(): 
    # I haven't been able to find a good way to test the p-value calculation in _fit_tail()
    # Here we test some properties of the other outputs 

    # ==========================================
    # 1. Analytic Derivation Tests
    # ==========================================

    def test_analytic_tail_midpoint(self):
        """
        Test standard tail thresholding (tailstartidx > 0) against hand-calculated values.
        
        Analytic Derivation:
        - Gdist = [2, 4, 6, 8, 10], tailstartidx = 2 (starts at 6.0)
        - upar = (6.0 + 4.0) / 2 = 5.0
        - ytail = [1.0, 3.0, 5.0]
        - x (mean) = 3.0
        - s2 (var) = ( (-2)^2 + 0^2 + 2^2 ) / 2 = 4.0
        - apar = 3.0 * (9.0 / 4.0 + 1) / 2 = 4.875
        - kpar = -(9.0 / 4.0 - 1) / 2 = -0.625
        """
        Gdist = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        
        _, apar, kpar, upar = _fit_tail(Gdist, tailstartidx=2)
        
        np.testing.assert_allclose(upar, 5.0)
        np.testing.assert_allclose(apar, 4.875)
        np.testing.assert_allclose(kpar, -0.625)

    def test_analytic_tail_boundary(self):
        """
        Test boundary extrapolation (tailstartidx == 0) against hand-calculated values.
        
        Analytic Derivation:
        - Gdist = [2, 4, 6, 8, 10], tailstartidx = 0
        - upar = (3 * 2.0 - 4.0) / 2 = 1.0
        - ytail = [1.0, 3.0, 5.0, 7.0, 9.0]
        - x (mean) = 5.0
        - s2 (var) = 10.0
        - apar = 5.0 * (25.0 / 10.0 + 1) / 2 = 8.75
        - kpar = -(25.0 / 10.0 - 1) / 2 = -0.75
        """
        Gdist = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        
        _, apar, kpar, upar = _fit_tail(Gdist, tailstartidx=0)
        
        np.testing.assert_allclose(upar, 1.0)
        np.testing.assert_allclose(apar, 8.75)
        np.testing.assert_allclose(kpar, -0.75)


    # ==========================================
    # 2. Descending vs Ascending Parity
    # ==========================================

    def test_descending_parity(self):
        """
        Test that descending arrays with descending=True produce the exact 
        same MoM parameters as an ascending array of the same steps.
        """
        Gdist_asc = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        Gdist_desc = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        
        # Call ascending (descending=False by default)
        A2_asc, apar_asc, kpar_asc, upar_asc = _fit_tail(Gdist_asc, tailstartidx=0)
        
        # Call descending with explicit descending=True
        A2_desc, apar_desc, kpar_desc, upar_desc = _fit_tail(Gdist_desc, tailstartidx=0)
        
        # 1. Verify upar calculations
        np.testing.assert_allclose(upar_asc, 1.0)
        np.testing.assert_allclose(upar_desc, 11.0)
        
        # 2. Shape and scale parameters must be identical
        np.testing.assert_allclose(apar_asc, apar_desc)
        np.testing.assert_allclose(kpar_asc, kpar_desc)

        # 3. Anderson-Darling p-values must match
        np.testing.assert_allclose(A2_asc, A2_desc)


    # ==========================================
    # 3. Distribution Scaling Invariance
    # ==========================================

    def test_shift_and_scale_invariance(self):
        """
        Test mathematically required properties of GPD parameters:
        1. Adding a constant (shift) leaves scale (apar) and shape (kpar) unchanged.
        2. Multiplying by a constant (scale) scales `apar` but leaves `kpar` unchanged.
        """
        base_dist = np.geomspace(1, 100, 100)
        
        _, apar_base, kpar_base, upar_base = _fit_tail(base_dist, tailstartidx=10)
        
        # Test 1: Shift by 100
        shift = 100.0
        _, apar_shift, kpar_shift, upar_shift = _fit_tail(base_dist + shift, tailstartidx=10)
        
        np.testing.assert_allclose(upar_shift, upar_base + shift)
        np.testing.assert_allclose(apar_shift, apar_base)
        np.testing.assert_allclose(kpar_shift, kpar_base)
        
        # Test 2: Scale by 5
        scale = 5.0
        _, apar_scale, kpar_scale, upar_scale = _fit_tail(base_dist * scale, tailstartidx=10)
        
        np.testing.assert_allclose(upar_scale, upar_base * scale)
        np.testing.assert_allclose(apar_scale, apar_base * scale)
        np.testing.assert_allclose(kpar_scale, kpar_base)


class Test_pareto(): 
    @pytest.mark.parametrize("descending", [False, True])
    @pytest.mark.parametrize("factor", [1, -1])
    @pytest.mark.parametrize("ndist", [50, 100, 1000, 10000])
    def test_no_refinement(self, descending, factor, ndist): 
        Gdist = factor * np.geomspace(1.0, 100.0, ndist)
        G = factor * np.array([30, 50, 70]) # nowhere near the tail :)

        P_emp = palm_datapval(G, Gdist, descending=descending)
        P, apar, kpar, upar = palm_pareto(G, Gdist, descending=descending, Pthr = 0.05)

        assert np.isnan(apar)
        assert np.isnan(kpar)
        assert np.isnan(upar)
        np.testing.assert_allclose(P, P_emp)

    @pytest.mark.parametrize("descending", [False, True])
    @pytest.mark.parametrize("factor", [1, -1])
    @pytest.mark.parametrize("ndist", [50, 100, 1000, 10000])
    def test_some_refinement(self, descending, factor, ndist): 

        Gdist = factor * np.geomspace(1.0, 100.0, ndist)
        G = factor * 85

        P_emp = palm_datapval(G, Gdist, descending=descending)
        P, apar, kpar, upar = palm_pareto(G, Gdist, descending=descending, Pthr = 0.05)

        if descending ^ (factor == 1): # Fitting should take place
            # G = +/-90 is in the extreme tail threshold (P_emp <= Pthr = 0.05)
            assert not np.isnan(apar), f"Expected valid 'apar' fit for descending={descending}, factor={factor}"
            assert not np.isnan(kpar), f"Expected valid 'kpar' fit for descending={descending}, factor={factor}"
            assert not np.isnan(upar), f"Expected valid 'upar' fit for descending={descending}, factor={factor}"
            assert 0.0 <= P <= 0.05
        else: # Fitting shouldn't take place
            # G = +/-90 sits in the non-extreme bulk (P_emp > Pthr = 0.05)
            assert np.isnan(apar), f"Expected NaN 'apar' for non-extreme G (descending={descending}, factor={factor})"
            assert np.isnan(kpar), f"Expected NaN 'kpar' for non-extreme G (descending={descending}, factor={factor})"
            assert np.isnan(upar), f"Expected NaN 'upar' for non-extreme G (descending={descending}, factor={factor})"
            np.testing.assert_allclose(P, P_emp)

    @pytest.mark.parametrize("descending_factor", [[False, 1], [True, -1]])
    @pytest.mark.parametrize("ndist", [100, 1000, 10000])
    @pytest.mark.parametrize("threshold", [0.9, 0.95])
    def test_refinement(self, descending_factor, ndist, threshold):
        """
        Test tail fitting and p-value refinement using geometrically spaced data 
        (ensures kpar stays safely within GPD table bounds).
        """

        descending, factor = descending_factor

        Gdist = factor * np.geomspace(1.0, 100.0, ndist)
        G = factor * 85
        
        P_emp = palm_datapval(G, Gdist, descending=descending)
        
        P, apar, kpar, upar = palm_pareto(
            G, Gdist, 
            descending=descending, 
            Pthr=0.05, 
            thresholds=threshold
        )
        
        # 1. Parameter validity checks
        assert not np.isnan(apar)
        assert not np.isnan(kpar)
        assert not np.isnan(upar)
        
        # 2. Check upar threshold calculation
        expected_upar = (Gdist[int(threshold*ndist-2)] + Gdist[int(threshold*ndist-1)]) / 2.0
        np.testing.assert_allclose(upar, expected_upar)

        # 3. Check that P value is close but not the same
        assert 0 <= P <= 0.05
        assert not np.isclose(P, P_emp), f"Expected refined P value to differ from empirical P value: P={P}, P_emp={P_emp}"
        np.testing.assert_allclose(P, P_emp, rtol=0.1)  

    @pytest.mark.parametrize("thresholds", [0.9, 0.95, [0.5, 0.75, 0.9, 0.95, 0.99]])
    @pytest.mark.parametrize("ndist", [100, 1000, 5000])
    def test_parity(self, thresholds, ndist):
        """
        Test that descending statistics yield identical GPD scale/shape parameters
        when the data are flipped.
        """
        Gdist_a = np.geomspace(1.0, 100.0, ndist)
        G_a = 90
        
        Gdist_d = -Gdist_a
        G_d = -G_a
        
        P_a, apar_a, kpar_a, upar_a = palm_pareto(
            G_a, Gdist_a, descending=False, thresholds=thresholds
        )
        P_d, apar_d, kpar_d, upar_d = palm_pareto(
            G_d, Gdist_d, descending=True,  thresholds=thresholds
        )

        np.testing.assert_allclose(P_a, P_d, equal_nan=False)
        np.testing.assert_allclose(apar_a, apar_d, equal_nan=False)
        np.testing.assert_allclose(kpar_a, kpar_d, equal_nan=False)
        np.testing.assert_allclose(upar_a, -upar_d, equal_nan=False)


    def test_palm_pareto_partial_pvalue_masking(self):
        """
        Test array of statistics G where only elements passing BOTH Pthr AND
        falling strictly inside the tail boundary (diffG > 0) are updated.
        """
        Gdist = np.geomspace(1.0, 100.0, 100)

        G = np.array([1.05, 50, 90])
        
        P_emp = palm_datapval(G, Gdist, descending=False)
        
        P, apar, kpar, upar = palm_pareto(
            G, Gdist, 
            descending=False, 
            Pthr=0.05
        )
        
        # Only G[0] should be updated by GPD
        assert P[0] == P_emp[0]
        assert P[1] == P_emp[1]
        assert P[2] != P_emp[2]

