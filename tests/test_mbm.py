import numpy as np
from scipy.stats import genpareto, kstest
import pytest

from neuromodes.mbm import (permutations_shuffle_rows, permutations_flip_sign, 
                            mbm_generate_stat_maps, mbm_calc_p_vals)


class TestPermutations:
    @pytest.mark.parametrize("function", [permutations_shuffle_rows, permutations_flip_sign])
    @pytest.mark.parametrize("seed", [None, 1, 42])
    @pytest.mark.parametrize("n_subjects", [2, 5, 10])
    @pytest.mark.parametrize("n_permutations", [5, 10])
    def test_permutation_shape(self, function, seed, n_subjects, n_permutations):
        """Check that the permutation matrix has the correct shape."""
        perm = function(
            seed=seed,
            n_subjects=n_subjects,
            n_permutations=n_permutations,
        )

        assert perm.shape == (n_subjects, n_permutations)

    @pytest.mark.parametrize("seed", [None, 1, 42])
    @pytest.mark.parametrize("n_subjects", [2, 5, 10])
    @pytest.mark.parametrize("n_permutations", [5, 10])
    def test_shuffle_rows_contents(self, seed, n_subjects, n_permutations):
        """Check that the permutation matrix has the correct contents."""
        perm = permutations_shuffle_rows(
            seed=seed,
            n_subjects=n_subjects,
            n_permutations=n_permutations,
        )

        for ii in range(n_permutations):
            assert set(perm[:, ii]) == set(range(n_subjects))

    @pytest.mark.parametrize("seed", [None, 1, 42])
    @pytest.mark.parametrize("n_subjects", [2, 5, 10])
    @pytest.mark.parametrize("n_permutations", [5, 10])
    def test_flip_sign_contents(self, seed, n_subjects, n_permutations):
        """Check that the permutation matrix has the correct contents."""
        perm = permutations_flip_sign(
            seed=seed,
            n_subjects=n_subjects,
            n_permutations=n_permutations,
        )

        assert np.isin(perm, [-1, 1]).all()

    @pytest.mark.parametrize("function", [permutations_shuffle_rows, permutations_flip_sign])
    @pytest.mark.parametrize("seed", range(20))
    @pytest.mark.parametrize("n_subjects", [2, 5, 10])
    @pytest.mark.parametrize("n_permutations", [5, 10])
    def test_permutation_n_permutations_stable(self, function, seed, n_subjects, n_permutations):
        """Check that the first n_permutations of a larger permutation set are the same as a smaller permutation set."""
        first = function(
            seed=seed,
            n_subjects=n_subjects,
            n_permutations=n_permutations,
        )

        second = function(
            seed=seed,
            n_subjects=n_subjects,
            n_permutations=n_permutations+1,
        )

        assert np.array_equal(first, second[:, :n_permutations])

    @pytest.mark.parametrize("seed", range(20))
    @pytest.mark.parametrize("n_subjects", [2, 5, 10])
    @pytest.mark.parametrize("n_permutations", [5, 10])
    def test_flip_sign_n_subjects_stable(self, seed, n_subjects, n_permutations):
        """Check that the first n_subjects of a larger flip-sign permutation set are the same as a smaller flip-sign permutation set."""
        first = permutations_flip_sign(
            seed=seed,
            n_subjects=n_subjects,
            n_permutations=n_permutations,
        )

        second = permutations_flip_sign(
            seed=seed,
            n_subjects=n_subjects+1,
            n_permutations=n_permutations,
        )

        assert np.array_equal(first, second[:n_subjects, :])

    @pytest.mark.parametrize("function", [permutations_shuffle_rows, permutations_flip_sign])
    @pytest.mark.parametrize("seeds", [[1, 2], [42, 43]])
    @pytest.mark.parametrize("n_subjects", [2, 5, 10])
    @pytest.mark.parametrize("n_permutations", [5, 10])
    def test_permutation_different_seeds(self, function, seeds, n_subjects, n_permutations):
        """Check that different seeds produce different permutation matrices."""
        perm1 = function(
            seed=seeds[0],
            n_subjects=n_subjects,
            n_permutations=n_permutations,
        )

        perm2 = function(
            seed=seeds[1],
            n_subjects=n_subjects,
            n_permutations=n_permutations,
        )

        assert not np.array_equal(perm1, perm2)

    @pytest.mark.parametrize("seed", range(1, 50)) # has to be a fixed set which is known to be within error bounds
    @pytest.mark.parametrize("n_subjects", [2, 5])
    @pytest.mark.parametrize("n_permutations", [2500]) # has to be large
    def test_shuffle_rows_distribution(self, seed, n_subjects, n_permutations):
        """Check that the distribution of shuffle-row permutations is approximately uniform."""
        perm = permutations_shuffle_rows(
            seed=seed,
            n_subjects=n_subjects,
            n_permutations=n_permutations,
        )

        rtol = 4 * np.sqrt((n_subjects-1) / n_permutations)  # ~4 SDs of a binomial distribution

        for ii in range(n_subjects): # check each row
            counts = np.bincount(perm[ii, :], minlength=n_subjects)
            expected_count = n_permutations / n_subjects
            np.testing.assert_allclose(counts, expected_count, rtol=rtol)

    @pytest.mark.parametrize("seed", range(1, 50)) # has to be a fixed set which is known to be within error bounds
    @pytest.mark.parametrize("n_subjects", [2, 5])
    @pytest.mark.parametrize("n_permutations", [2500]) # has to be large
    def test_flip_sign_distribution(self, seed, n_subjects, n_permutations):
        """Check that the distribution of flip-sign permutations is approximately uniform."""
        perm = permutations_flip_sign(
            seed=seed,
            n_subjects=n_subjects,
            n_permutations=n_permutations,
        )

        np.testing.assert_allclose(np.mean(perm, axis=1), 0, atol=0.1)  # check each subject (row)
        np.testing.assert_allclose(np.mean(perm), 0, atol=0.05)         # check whole matrix


class TestStatMaps(): 
    @pytest.fixture(scope="class")
    def glm_data(self):
        """Generates a reproducible 2-predictor GLM setup with a clear signal."""
        rng = np.random.default_rng(42)
        n_subjects = 50
        n_data = 30

        # Continuous target predictor + continuous nuisance predictor
        x_target = rng.standard_normal((n_subjects, 1))
        x_nuisance = rng.standard_normal((n_subjects, 1))
        design = np.hstack([x_target, x_nuisance])

        # Response with strong effect on predictor 0
        signal = x_target * 2.5
        noise = rng.standard_normal((n_subjects, n_data))
        response = signal + noise

        contrast_1row = np.array([[1.0, 0.0]])
        contrast_2row = np.array([[1.0, 0.0], [0.0, 1.0]])

        return {
            "design": design,
            "response": response,
            "contrast_1row": contrast_1row,
            "contrast_2row": contrast_2row,
            "n_subjects": n_subjects,
            "n_data": n_data,
        }

    @pytest.mark.parametrize("statTest", ["ttest", "ztest", "ftest"])
    def test_empirical_shapes_and_types(self, glm_data, statTest):
        stat_map = mbm_generate_stat_maps(
            glm_data["design"],
            glm_data["response"],
            glm_data["contrast_1row"],
            statTest,
        )

        assert isinstance(stat_map, np.ndarray)
        assert np.issubdtype(stat_map.dtype, np.floating)
        assert stat_map.shape == (glm_data["n_data"],)
        assert not np.isnan(stat_map).any()

    @pytest.mark.parametrize("statTest", ["ttest", "ztest", "ftest"])
    @pytest.mark.parametrize("permutationMethod", ["flip_sign", "shuffle_rows"])
    @pytest.mark.parametrize("n_permutations", [5, 10])
    def test_nulls_1row_shapes_and_types(self, glm_data, statTest, permutationMethod, n_permutations):
        null_maps = mbm_generate_stat_maps(
            glm_data["design"],
            glm_data["response"],
            glm_data["contrast_1row"],
            statTest=statTest,
            permutationMethod=permutationMethod,
            n_permutations=n_permutations,
            seed=42,
        )

        assert isinstance(null_maps, np.ndarray)
        assert np.issubdtype(null_maps.dtype, np.floating)
        assert null_maps.shape == (glm_data["n_data"], n_permutations)
        assert not np.isnan(null_maps).any()

    @pytest.mark.parametrize("statTest", ["ftest"]) # multirow only for F-test
    @pytest.mark.parametrize("permutationMethod", ["flip_sign", "shuffle_rows"])
    @pytest.mark.parametrize("n_permutations", [5, 10])
    def test_nulls_2row_shapes_and_types(self, glm_data, statTest, permutationMethod, n_permutations):
        null_maps = mbm_generate_stat_maps(
            glm_data["design"],
            glm_data["response"],
            glm_data["contrast_2row"],
            statTest=statTest,
            permutationMethod=permutationMethod,
            n_permutations=n_permutations,
            seed=42,
        )

        assert isinstance(null_maps, np.ndarray)
        assert np.issubdtype(null_maps.dtype, np.floating)
        assert null_maps.shape == (glm_data["n_data"], n_permutations)
        assert not np.isnan(null_maps).any()


    @pytest.mark.parametrize("statTest", ["ttest", "ztest", "ftest"])
    @pytest.mark.parametrize("permutationMethod", ["flip_sign", "shuffle_rows"])
    @pytest.mark.parametrize("n_permutations", [1000]) # has to be large
    @pytest.mark.parametrize("seed", range(3)) # has to be a fixed set which is known to be within error bounds
    def test_permutation_p_values_recover_signal(self, glm_data, statTest, permutationMethod, n_permutations, seed):
        """Verifies that permutation tests yield significant p-values for true signals
        and non-significant p-values for noise/nuisance.
        """
        design = glm_data["design"]
        response = glm_data["response"]

        # --- 1. Test True Signal (Predictor 0) ---
        contrast_signal = np.array([[1.0, 0.0]])
        obs_stat_signal = mbm_generate_stat_maps(
            design, 
            response, 
            contrast_signal, 
            statTest=statTest
        )
        null_stats_signal = mbm_generate_stat_maps(
            design,
            response,
            contrast_signal,
            statTest=statTest,
            permutationMethod=permutationMethod,
            n_permutations=n_permutations,
            seed=seed,
        )

        # The mean of the null t-statistics across permutations 
        # should be close to 0 for ttest and ztest (F-test is not centred at 0)
        if statTest in ["ttest", "ztest"]:
            null_means = np.mean(null_stats_signal, axis=1)
            np.testing.assert_allclose(null_means, 0.0, atol=0.25)
            
        # Empirical p-values: proportion of null stats >= observed stat
        # Signal was strong (2.5), so p-values should be near minimal (1 / (n_perm + 1))
        p_vals_signal = np.mean(
            null_stats_signal >= obs_stat_signal[:, None], axis=1
        )
        assert np.all(p_vals_signal < 0.05)


    @pytest.mark.parametrize("statTest", ["ttest", "ztest", "ftest"])
    @pytest.mark.parametrize("permutationMethod", ["flip_sign", "shuffle_rows"])
    @pytest.mark.parametrize("n_permutations", [100])
    @pytest.mark.parametrize("seed", range(3)) # has to be a fixed set which is known to be within error bounds
    def test_permutation_p_values_recover_nuisance(self, glm_data, statTest, permutationMethod, n_permutations, seed):
        """Verifies that permutation tests yield significant p-values for true signals
        and non-significant p-values for noise/nuisance.
        """
        design = glm_data["design"]
        response = glm_data["response"]
        # --- 2. Test Pure Nuisance (Predictor 1, no injected effect) ---
        contrast_nuisance = np.array([[0.0, 1.0]])
        obs_stat_nuisance = mbm_generate_stat_maps(
            design, 
            response, 
            contrast_nuisance, 
            statTest=statTest
        )
        null_stats_nuisance = mbm_generate_stat_maps(
            design,
            response,
            contrast_nuisance,
            statTest=statTest,
            permutationMethod=permutationMethod,
            n_permutations=n_permutations,
            seed=seed,
        )

        p_vals_nuisance = np.mean(
            null_stats_nuisance >= obs_stat_nuisance[:, None], axis=1
        )

        # Nuisance predictor has no signal -> p-values should follow ~Uniform(0, 1)
        assert np.mean(p_vals_nuisance) > 0.25
        assert np.mean(kstest(p_vals_nuisance, 'uniform').pvalue) > 0.1  # Kolmogorov-Smirnov test for uniformity


class TestPVals(): 

    # Test simple cases without GPD fitting
    @pytest.mark.parametrize("n_data", [10, 50])
    @pytest.mark.parametrize("n_permutations", [100, 1000])
    def test_no_gpd(self, n_data, n_permutations):
        observed = np.random.randn(n_data)
        nulls = np.random.randn(n_data, n_permutations)

        p_geq = np.mean(nulls >= observed[:, None], axis=1)
        p_leq = np.mean(nulls <= observed[:, None], axis=1)
        p_two = 2.0 * np.minimum(p_geq, p_leq)

        # Test the different alternatives
        tests = {
            'greater': p_geq,
            'less': p_leq,
            'two-sided': p_two
        }
        
        for alternative, expected in tests.items():
            p_act = mbm_calc_p_vals(observed, nulls, alternative,   # type: ignore
                                   n_for_tail_estimation=(),        # turn off warnings
                                   n_for_pval_recalculation=0)      # turn off fitting

            np.testing.assert_array_equal(expected, p_act)
       
    @pytest.mark.parametrize("c", [0.1, 0.5, 0.85])
    @pytest.mark.parametrize("scale", [1.0, 2.0])
    @pytest.mark.parametrize("loc", [0.0, 1.0])
    @pytest.mark.parametrize("n_permutations", [1000, 10000])
    @pytest.mark.parametrize("n_for_tail_estimation", [range(1000, 1, -10)])
    def test_gpd_fit_near_exact(self, c, scale, loc, n_permutations, n_for_tail_estimation):
        """Use exact GPD distribution to generate tails and verify the extrapolation."""
        
        # 1. Uniformly spaced quantile midpoints over N=n_permutations
        # Midpoints ensure no point maps to exact CDF boundary 0.0 or 1.0
        p_grid = (np.arange(n_permutations)+0.5) / (n_permutations)
    
        # 2. Generate exact sample grid
        null_distribution = genpareto.ppf(p_grid, c, loc=loc, scale=scale)
    
        # 3. Target observed statistic at target p = 0.0005
        p_target = (0.25) / (n_permutations)
        observed_stat = genpareto.ppf(1 - p_target, c, loc=loc, scale=scale)
    
        observedStatMap = np.array([observed_stat])
        statMapNull = null_distribution.reshape(1, -1)
    
        # 4. Check naive empirical p-value
        p_naive = mbm_calc_p_vals(
            observedStatMap=observedStatMap,
            statMapNull=statMapNull,
            alternative="greater",
            n_for_pval_recalculation=0
        )
        assert p_naive == 0.0
    
        # 5. Call function with correct range direction (-10 step)
        p_fit = mbm_calc_p_vals(
            observedStatMap=observedStatMap,
            statMapNull=statMapNull,
            alternative="greater",
            n_for_tail_estimation=n_for_tail_estimation,
            n_for_pval_recalculation=10,
            gof_acceptance_level=0.9, # could even be 1 as the data are from the exact distribution
            return_tail=False,
        )
    
        # Verify accuracy against target_p
        np.testing.assert_allclose(
            p_fit,
            p_target,
            rtol=1e-1,
            err_msg=f"Expected interpolated p-value to be close to {p_target}, got {p_fit}",
        )

    # Test fitting procedure but not GOF results (set to 0 here)
    @pytest.mark.parametrize("n_null", [10000])
    @pytest.mark.parametrize("n_tail", [1000]) # needs to always be less than n_null
    @pytest.mark.parametrize("c", [-0.45, 0, 0.85])
    @pytest.mark.parametrize("loc", [12.0])
    @pytest.mark.parametrize("scale", [2.0])
    @pytest.mark.parametrize("p", [1e-4, 1e-5, 1e-6])
    @pytest.mark.parametrize("alternative_and_factor", [("greater", 1.0), ("less", -1.0)])
    @pytest.mark.parametrize("seed", [42])
    def test_all_gpd_one_sided(self, n_null, n_tail, c, loc, scale, p, alternative_and_factor, seed):
        """Test that the fitting finds a more accurate p-value. Note that the GOF thresholding is not tested here."""
        alternative, factor = alternative_and_factor
        obs = factor * genpareto.ppf(1-p, c, loc=loc, scale=scale)
        nulls = factor *  np.sort(genpareto.rvs(c, loc=loc, scale=scale, size=(1,n_null), random_state=seed))

        floc = np.mean(factor * nulls[:,-n_tail-1:-n_tail+1])
        fit = genpareto.fit(factor * nulls[:,-n_tail:] - floc, floc = 0.0)
        p_exp = genpareto.sf(factor * obs - floc, *fit) * n_tail/n_null

        p_test = mbm_calc_p_vals(np.array([obs]), 
                             nulls,
                             alternative=alternative, 
                             n_for_tail_estimation=n_tail,
                             n_for_pval_recalculation=np.inf, 
                             gof_acceptance_level=0.00,
                             )

        p_naive = mbm_calc_p_vals(np.array([obs]), 
                            nulls,
                            alternative=alternative, 
                            n_for_tail_estimation=n_tail,
                            n_for_pval_recalculation=0, # no recalculation
                             )
     
        print(f"p: {p}, p_exp: {p_exp}, p_test: {p_test}, p_naive: {p_naive}")
        np.testing.assert_allclose(p_test, p, rtol=1, atol=0) # has to be pretty loose here
        np.testing.assert_allclose(p_test, p_exp, rtol=1e-6, atol=1e-6)
        assert p_naive!=p_test or p_test==0, "P-values with and without fitting should be different."

    @pytest.mark.parametrize("n_tail", [50, 100])
    @pytest.mark.parametrize("p", [1e-4, 1e-6])
    @pytest.mark.parametrize("seed", range(24)) # has to be a fixed set which is known to be within error bounds (false negs of AD test)
    def test_gpd_tail_fallback_selection(self, n_tail, p,seed):
        """Verify that a large invalid tail fails GoF and falls back to a valid smaller tail."""
        
        # Bulk data (Normal) + Extreme Tail (GPD), 99% of data is bulk
        bulk = 3.0 - np.random.default_rng(seed).exponential(scale=1.0, size=(1, n_tail * 99))
        tail = genpareto.rvs(c=0.2, loc=3.0, scale=1.0, size=(1, n_tail), random_state=seed)
        nulls = np.sort(np.hstack([bulk, tail]), axis=1)

        obs = np.array([genpareto.ppf(1-p, c=0.2, loc=3.0, scale=1.0)])

        # Call 1
        p_test = mbm_calc_p_vals(
            obs, nulls,
            alternative='greater',
            n_for_tail_estimation=(10*n_tail, n_tail),
            n_for_pval_recalculation=np.inf, # force recalculation
            gof_acceptance_level=0.05 # check that first is rejected
        )

        # Call 2
        p_good = mbm_calc_p_vals(
            obs, nulls,
            alternative='greater',
            n_for_tail_estimation=(n_tail,),
            n_for_pval_recalculation=np.inf,
            gof_acceptance_level=0.05
        )

        # Call 3
        p_bad = mbm_calc_p_vals(
            obs, nulls,
            alternative='greater',
            n_for_tail_estimation=(10*n_tail,),
            n_for_pval_recalculation=np.inf,
            gof_acceptance_level=0.00 # force this to be accepted here so that we can get a value
        )

        p_naive = mbm_calc_p_vals(
            obs, nulls,
            alternative='greater',
            n_for_pval_recalculation=0, # no recalculation
        )

        print(f"p: {p}, p_test: {p_test}, p_good: {p_good}, p_bad: {p_bad}, p_naive: {p_naive}")

        # If fallback succeeded
        np.testing.assert_allclose(p_test, p_good, rtol=1e-6, atol=1e-6)
        
        # Confirm that nte=1000 yields a different p-value than nte=10
        assert not np.isclose(p_test, p_bad, rtol=1e-3, atol=0.0) or np.isclose(p_good, 0.0, atol=1e-6) 
        assert not np.isclose(p_test, p_naive, rtol=1e-3, atol=0.0) or np.isclose(p_good, 0.0, atol=1e-6)


    # Add tests to:
    # - for the goodness of fit testing (which is off by default)
    # - check that revmap is correct
    # - when n_for_tail_estimation is a list, check that the appropriate fit is used
    # - that the 'continue' statements prevent bad behaviour on badly conditioned data
    # 

