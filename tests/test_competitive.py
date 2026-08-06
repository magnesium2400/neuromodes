import pytest
import numpy as np
from neuromodes.palm import palm_competitive

@pytest.mark.parametrize("X", [np.array([10, 20, 20, 30])])
@pytest.mark.parametrize("ord_val, mod_val, exp_unsrtR, exp_S, exp_srtR", [
    # (ord, mod, expected_unsrtR, expected_S, expected_srtR)
    ('ascend',  False, [1, 2, 2, 4], [10, 20, 20, 30], [1, 2, 2, 4]),
    ('ascend',  True,  [1, 3, 3, 4], [10, 20, 20, 30], [1, 3, 3, 4]),
    ('descend', False, [4, 2, 2, 1], [30, 20, 20, 10], [1, 2, 2, 4]),
    ('descend', True,  [4, 3, 3, 1], [30, 20, 20, 10], [1, 3, 3, 4]),
])
def test_ranking_logic_full_1d(X, ord_val, mod_val, exp_unsrtR, exp_S, exp_srtR):
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
    
    unsrtR, S, srtR = palm_competitive(X, ord=ord_val, mod=mod_val)
    
    np.testing.assert_array_equal(unsrtR, np.array(exp_unsrtR))
    np.testing.assert_array_equal(S, np.array(exp_S))
    np.testing.assert_array_equal(srtR, np.array(exp_srtR))


def test_multidimensional():
    """Test functionality with 2D arrays."""
    X = np.array([[1, 3], [2, 2], [2, 1]])*10
    unsrtR, _, _ = palm_competitive(X, ord='ascend', mod=False)
    # Col 1: 1(1), 2(2), 2(2) -> [1, 2, 2]
    # Col 2: 3(3), 2(2), 1(1) -> [3, 2, 1]
    expected = np.array([[1, 3], [2, 2], [2, 1]])
    np.testing.assert_array_equal(unsrtR, expected)

def test_infinity_handling():
    """Test that function handles infinities without crashing."""
    X = np.array([1, np.inf, 2])
    unsrtR, _, _ = palm_competitive(X, ord='ascend', mod=False)
    # 1(1), 2(2), inf(3)
    expected = np.array([1, 3, 2])
    np.testing.assert_array_equal(unsrtR, expected)

def test_invalid_data():
    """Test that NaNs raise a ValueError."""
    X = np.array([1, np.nan, 3])
    with pytest.raises(ValueError, match="Check for NaNs"):
        palm_competitive(X)
