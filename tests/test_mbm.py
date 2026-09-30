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
    def test_no_gpd_greater(self, n_data, n_permutations):
        observed = np.random.randn(n_data)
        nulls = np.random.randn(n_data, n_permutations)
        
        pexp = np.mean(nulls >= observed[:, None], axis=1)
        pact = mbm_calc_p_vals(observed, nulls, 'greater', 
                               n_for_tail_estimation=(),    # turn off warnings
                               n_for_pval_recalculation=0)  # turn off fitting

        np.testing.assert_array_equal(pexp, pact)

    @pytest.mark.parametrize("n_data", [10, 50])
    @pytest.mark.parametrize("n_permutations", [100, 1000])
    def test_no_gpd_less(self, n_data, n_permutations):
        observed = np.random.randn(n_data)
        nulls = np.random.randn(n_data, n_permutations)
        
        pexp = np.mean(nulls <= observed[:, None], axis=1)
        pact = mbm_calc_p_vals(observed, nulls, 'less', 
                               n_for_tail_estimation=(), 
                               n_for_pval_recalculation=0)

        np.testing.assert_array_equal(pexp, pact)

    @pytest.mark.parametrize("n_data", [10, 50])
    @pytest.mark.parametrize("n_permutations", [100, 1000])
    def test_no_gpd_two_sided(self, n_data, n_permutations):
        observed = np.random.randn(n_data)
        nulls = np.random.randn(n_data, n_permutations)
        
        pexp = 2.0 * np.minimum(np.mean(nulls >= observed[:, None], axis=1), 
                                np.mean(nulls <= observed[:, None], axis=1))
        pact = mbm_calc_p_vals(observed, nulls, 'two-sided', 
                               n_for_tail_estimation=(), 
                               n_for_pval_recalculation=0)

        np.testing.assert_array_equal(pexp, pact)


    # Test some cases where GPD is always fit (manually redo GPD fitting)
    @pytest.mark.parametrize("n_null", [42, 240])
    @pytest.mark.parametrize("n_tail", [10, 24]) # needs to always be less than n_null
    @pytest.mark.parametrize("c", [-0.1, 0.0, 0.1])
    @pytest.mark.parametrize("loc", [0.0, 1.0])
    @pytest.mark.parametrize("scale", [1.0, 2.0])
    @pytest.mark.parametrize("p", [0.001, 0.01])
    def test_all_gpd_greater(self, n_null, n_tail, c, loc, scale, p):
        obs = genpareto.ppf(1-p, c, loc=loc, scale=scale)
        nulls = np.sort(genpareto.rvs(c, loc=loc, scale=scale, size=(1,n_null)))

        floc = np.mean(nulls[:,-n_tail-1:-n_tail+1])
        fit = genpareto.fit(nulls[:,-n_tail:], floc = floc)
        p1 = genpareto.sf(obs, *fit) * n_tail/n_null

        p2 = mbm_calc_p_vals(np.array([obs]), 
                             nulls,
                             alternative='greater', 
                             n_for_tail_estimation=n_tail,
                             n_for_pval_recalculation=np.inf, 
                             )

        np.testing.assert_allclose(p1, p2, rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("n_null", [42, 240])
    @pytest.mark.parametrize("n_tail", [10, 24]) # needs to always be less than n_null
    @pytest.mark.parametrize("c", [-0.1, 0.0, 0.1])
    @pytest.mark.parametrize("loc", [0.0, 1.0])
    @pytest.mark.parametrize("scale", [1.0, 2.0])
    @pytest.mark.parametrize("p", [0.001, 0.01])
    def test_all_gpd_less(self, n_null, n_tail, c, loc, scale, p):
        obs = genpareto.ppf(p, c, loc=loc, scale=scale)
        nulls = -np.sort(-genpareto.rvs(c, loc=loc, scale=scale, size=(1, n_null)))

        # model on negative data
        floc = np.mean(-nulls[:,-n_tail-1:-n_tail+1])
        fit = genpareto.fit(-nulls[:,-n_tail:], floc = floc)
        p1 = genpareto.sf(-obs, *fit) * (n_tail / n_null)

        p2 = mbm_calc_p_vals(
            np.array([obs]), 
            nulls,
            alternative='less', 
            n_for_tail_estimation=n_tail,
            n_for_pval_recalculation=np.inf, 
        )

        np.testing.assert_allclose(p1, p2, rtol=1e-9, atol=1e-9)



    # Add tests to:
    # - check that revmap is correct
    # - explicitly check that fitting makes p-values more accurate
    # - when n_for_tail_estimation is a list, check that the appropriate fit is used
    # - that the 'continue' statements prevent bad behaviour on badly conditioned data
    # - for the goodness of fit testing (which is off by default)
    # 

