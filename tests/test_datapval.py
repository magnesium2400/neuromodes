import numpy as np
import pytest
from neuromodes.palm import palm_datapval

# =====================================================================
# 1. Basic Functionality Tests (Understandable & Core Usage)
# =====================================================================

def test_datapval_basic_usage1():
    """
    Demonstrates basic function behavior with standard inputs.
    
    Reference values: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] (N=10)
    Default (rev=False): Larger statistics are more significant.
    Reversed (rev=True): Smaller statistics are more significant.
    """
    gvals = np.arange(10, 110, 10)  # 10 reference permutation values
    g_test = np.array([15, 55, 95])

    # Case 1: Standard (rev=False)
    # G=15 -> 9/10 values >= 15 -> p = 0.9
    # G=55 -> 5/10 values >= 55 -> p = 0.5
    # G=95 -> 1/10 values >= 95 -> p = 0.1
    pvals_standard = palm_datapval(g_test, gvals, rev=False)
    expected_standard = np.array([0.9, 0.5, 0.1])
    np.testing.assert_allclose(pvals_standard, expected_standard)

    # Case 2: Reversed (rev=True)
    # G=15 -> 1/10 values <= 15 -> p = 0.1
    # G=55 -> 5/10 values <= 55 -> p = 0.5
    # G=95 -> 9/10 values <= 95 -> p = 0.9
    pvals_rev = palm_datapval(g_test, gvals, rev=True)
    expected_rev = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(pvals_rev, expected_rev)

def test_datapval_basic_usage2():
    """
    Demonstrates basic function behavior with standard inputs.
    
    Reference values: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] (N=10)
    Default (rev=False): Larger statistics are more significant.
    Reversed (rev=True): Smaller statistics are more significant.
    """
    gvals = np.arange(10, 110, 10)  # 10 reference permutation values
    g_test = np.array([110, 100, 50, 10, 0])

    # Case 1: Standard (rev=False)
    # G=110 -> 0/10 values >= 110 -> p = 0.0
    # G=100 -> 1/10 values >= 100 -> p = 0.1
    # G=50  -> 6/10 values >= 50  -> p = 0.6
    # G=10  -> 10/10 values >= 10 -> p = 1.0
    # G=0   -> 10/10 values >= 0  -> p = 1.0
    pvals_standard = palm_datapval(g_test, gvals, rev=False)
    expected_standard = np.array([0.0, 0.1, 0.6, 1.0, 1.0])
    np.testing.assert_allclose(pvals_standard, expected_standard)

    # Case 2: Reversed (rev=True)
    # G=0   -> 0/10 values <= 0   -> p = 0.0
    # G=10  -> 1/10 values <= 10  -> p = 0.1
    # G=50  -> 5/10 values <= 50  -> p = 0.5
    # G=100 -> 10/10 values <= 100 -> p = 1.0
    # G=110 -> 10/10 values <= 110 -> p = 1.0
    pvals_rev = palm_datapval(g_test, gvals, rev=True)
    expected_rev = np.array([1.0, 1.0, 0.5, 0.1, 0.0])
    np.testing.assert_allclose(pvals_rev, expected_rev)


# =====================================================================
# 2. Comprehensive & Edge Case Tests
# =====================================================================

@pytest.mark.parametrize("rev", [False, True])
def test_out_of_bounds_tails(rev):
    """Verify extreme tail behavior beyond min/max reference bounds."""
    gvals = np.array([10.0, 20.0, 30.0])

    if not rev:
        # G > max(gvals) -> p = 0.0 (more significant than all permutations)
        # G < min(gvals) -> p = 1.0 (less significant than all permutations)
        assert palm_datapval([100.0], gvals, rev=False)[0] == pytest.approx(0.0)
        assert palm_datapval([-50.0], gvals, rev=False)[0] == pytest.approx(1.0)
    else:
        # G < min(gvals) -> p = 0.0 (smaller/more significant than all)
        # G > max(gvals) -> p = 1.0 (larger/less significant than all)
        assert palm_datapval([-50.0], gvals, rev=True)[0] == pytest.approx(0.0)
        assert palm_datapval([100.0], gvals, rev=True)[0] == pytest.approx(1.0)

@pytest.mark.parametrize("gvals", [np.array([10, 50, 100])])
@pytest.mark.parametrize("rev, g_input, expected_pvals", [
    # rev = False (Right-inclusive: (E_i, E_i+1]):
    # Bin 0: (-inf, 10.0]  -> p = 1.0
    # Bin 1: (10.0, 50.0]  -> p = 2/3
    # Bin 2: (50.0, 100.0] -> p = 1/3
    # Bin 3: (100.0, inf)  -> p = 0.0
    (
        False, 
        [5.0, 9.9, 10.0, 10.1, 25.0, 49.9, 50.0, 50.1, 75.0, 99.9, 100.0, 100.1, 150.0], 
        [1.0, 1.0, 1.0,  2/3,  2/3,  2/3,  2/3,  1/3,  1/3,  1/3,   1/3,   0.0,   0.0]
    ),
    # rev = True (Left-inclusive: [E_i, E_i+1)):
    # Bin 0: (-inf, 10.0)  -> p = 0.0
    # Bin 1: [10.0, 50.0)  -> p = 1/3
    # Bin 2: [50.0, 100.0) -> p = 2/3
    # Bin 3: [100.0, inf)  -> p = 1.0
    (
        True,  
        [5.0, 9.9, 10.0, 10.1, 25.0, 49.9, 50.0, 50.1, 75.0, 99.9, 100.0, 100.1, 150.0], 
        [0.0, 0.0,  1/3,  1/3,  1/3,  1/3,  2/3,  2/3,  2/3,  2/3,   1.0,   1.0,   1.0]
    ),
])
def test_datapval_bin_inclusion(gvals, rev, g_input, expected_pvals):
    actual_pvals = palm_datapval(g_input, gvals, rev=rev)
    np.testing.assert_allclose(actual_pvals, expected_pvals)

@pytest.mark.parametrize("g_val, expected_pval", [
    (95, 0.1),  # (90, 100] -> p = 1/10 = 0.1
    (85, 0.2),  # (80, 90]  -> p = 2/10 = 0.2
    (15, 0.9),  # (10, 20]  -> p = 9/10 = 0.9
])
def test_unseen_intermediate_values_between_thresholds(g_val, expected_pval):
    """Test values falling strictly between observed permutation thresholds."""
    gvals = np.arange(10, 110, 10)
    res = palm_datapval([g_val], gvals, rev=False)
    assert res[0] == pytest.approx(expected_pval)


def test_tied_reference_values():
    """Verify correct empirical CDF step handling when reference data has ties."""
    # N = 5, values 20 appears three times
    gvals_tied = np.array([10, 20, 20, 20, 50])

    # rev=False: 4 out of 5 values are >= 20 -> p = 4/5 = 0.8
    pval_false = palm_datapval([20], gvals_tied, rev=False)
    assert pval_false[0] == pytest.approx(0.8)

    # rev=True: 4 out of 5 values are <= 20 -> p = 4/5 = 0.8
    pval_true = palm_datapval([20], gvals_tied, rev=True)
    assert pval_true[0] == pytest.approx(0.8)

def test_single_element_reference():
    """Edge case: Reference distribution has only 1 observation."""
    gvals = np.array([50.0])

    # Test values above, equal, and below single threshold
    res_false = palm_datapval([60, 50, 40], gvals, rev=False)
    np.testing.assert_allclose(res_false, [0.0, 1.0, 1.0])

    res_true = palm_datapval([40, 50, 60], gvals, rev=True)
    np.testing.assert_allclose(res_true, [0.0, 1.0, 1.0])

@pytest.mark.parametrize("shape", [(10,), (5, 2), (3, 4, 2)])
def test_array_shape_and_dimension_preservation(shape):
    """Output array shape must match input G shape exactly (1D, 2D, 3D)."""
    g = np.random.rand(*shape)
    res = palm_datapval(g, np.linspace(0, 1, 10))
    assert res.shape == shape

@pytest.mark.parametrize("rev", [False, True])
def test_pvalue_bounds_invariant(rev):
    """Property test: All resulting p-values must strictly lie in [0, 1]."""
    gvals = np.random.randn(100)
    g_random = np.random.randn(50) * 5.0  # Includes out-of-bounds numbers
    pvals = palm_datapval(g_random, gvals, rev=rev)
    assert np.all(pvals >= 0.0)
    assert np.all(pvals <= 1.0)


# =====================================================================
# 3. Input & Type Validation Tests
# =====================================================================

def test_handles_python_lists_and_scalars():
    """Function should accept Python lists/floats and convert them seamlessly."""
    gvals = [10, 20, 30, 40]
    
    # Passing scalar float
    p_scalar = palm_datapval(40, gvals)
    assert isinstance(p_scalar, (np.ndarray, float, np.floating))

    # Passing raw Python list
    p_list = palm_datapval([40, 10], gvals)
    np.testing.assert_allclose(p_list, [0.25, 1.0])


def test_empty_reference_gvals_raises_error():
    """Passing empty reference array Gvals must raise ValueError."""
    with pytest.raises(ValueError):
        palm_datapval([10, 20], Gvals=[])


def test_empty_g_input_returns_empty_array():
    """Passing empty G input should safely return an empty array."""
    res = palm_datapval([], [10, 20, 30])
    assert res.size == 0


# def test_nan_or_inf_in_g_input():
#     """Verify how NaNs or Infs in test array G propagate."""
#     gvals = np.array([10, 20, 30])
#     g_with_nan = np.array([20, np.nan, np.inf, -np.inf])

#     pvals = palm_datapval(g_with_nan, gvals, rev=False)

#     # Exact threshold
#     assert pvals[0] == pytest.approx(2/3)
#     # NaN propagation check
#     assert np.isnan(pvals[1])
#     # +Inf -> 0.0, -Inf -> 1.0
#     assert pvals[2] == pytest.approx(0.0)
#     assert pvals[3] == pytest.approx(1.0)


def test_invalid_rev_type_handling():
    """Verify non-boolean truthy/falsy coercion for 'rev' parameter."""
    gvals = np.array([10, 20, 30])
    
    # Integer 1 should coerce to True
    res_int_truthy = palm_datapval([10], gvals, rev=1)
    res_bool_true = palm_datapval([10], gvals, rev=True)
    np.testing.assert_array_equal(res_int_truthy, res_bool_true)

    # Integer 0 should coerce to False
    res_int_falsy = palm_datapval([10], gvals, rev=0)
    res_bool_false = palm_datapval([10], gvals, rev=False)
    np.testing.assert_array_equal(res_int_falsy, res_bool_false)


# def test_invalid_data_types_raise_typeerror():
#     """Non-numeric string elements in G or Gvals should raise TypeError/ValueError."""
#     with pytest.raises((TypeError, ValueError)):
#         palm_datapval(["invalid_string"], [10, 20, 30])

#     with pytest.raises((TypeError, ValueError)):
#         palm_datapval([10, 20], ["invalid_string"])