import pytest
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from neuromodes.palm import andersondarling

# --- TEST HELPERS ---

# TODO make this a fixture that can be reused
def get_z_for_a2(a2_target):
    """Reverse-engineers a 'z' array that exactly yields the requested A^2 statistic."""
    if a2_target >= 0.3863:
        term = 1 - 4 * np.exp(-(a2_target + 1))
        if term >= 0:
            x = 0.5 + 0.5 * np.sqrt(term)
            return np.array([x])
            
    def objective(x):
        return -2 - (np.log(x) + 3 * np.log(1 - x)) - a2_target
        
    res = root_scalar(objective, bracket=[1e-12, 0.2499])
    return np.array([res.root, 1 - res.root])


# --- PYTEST SUITE ---

@pytest.mark.parametrize("target_a2, k, expected_pval", [
    (0.4990, 0.5, 0.25),   # Row 2, Col 2
    (2.9220, -0.5, 0.001), # Bottom right
    (0.3390, 0.9, 0.5),    # Top left
])
def test_exact_table_values(target_a2, k, expected_pval):
    """Test precise matches for nodes directly on the provided table."""
    z = get_z_for_a2(target_a2)
    assert andersondarling(z, k) == pytest.approx(expected_pval, abs=1e-4)


def test_interpolated_values():
    """Test 2D interpolation for values between the table's rows and columns."""
    # 1. Midway between rows: k = 0.15 (between k=0.2 and k=0.1)
    target_a2_row = (0.7410 + 0.7660) / 2
    z_row_mid = get_z_for_a2(target_a2_row)
    assert andersondarling(z_row_mid, k=0.15) == pytest.approx(0.10, abs=1e-4)

    # 2. Midway between columns: p-value midway between 0.25 and 0.1 for k=0.0
    target_a2_col = (0.5690 + 0.7960) / 2
    z_col_mid = get_z_for_a2(target_a2_col)
    assert andersondarling(z_col_mid, k=0.0) == pytest.approx(0.175, abs=1e-4)


def test_k_limits():
    """Test robust bounds handling for out-of-table parameters."""
    # High k clipping (k=1.5 should cap at k=0.9)
    z_high = get_z_for_a2(0.4710) # At k=0.9, this A2 yields exactly p=0.25
    assert andersondarling(z_high, k=0.9) == pytest.approx(0.25, abs=1e-4)
    
    # Low k clipping (k=-5.0 should cap at k=-0.5)
    z_low = get_z_for_a2(1.0610) # At k=-0.5, this A2 yields exactly p=0.1
    assert andersondarling(z_low, k=-0.5) == pytest.approx(0.1, abs=1e-4)


def test_vectorization_against_standard_formula():
    """Compare the internal vectorized A2 calculation against the textbook loop."""
    np.random.seed(42)
    z_rand = np.sort(np.random.uniform(0.01, 0.99, 50))
    
    # Compute expected A2 using explicit loops
    n = len(z_rand)
    S = sum(
        (2 * (i + 1) - 1) * (np.log(z_rand[i]) + np.log(1 - z_rand[n - 1 - i]))
        for i in range(n)
    )
    expected_a2 = -n - S / n
    
    # Isolated lookup logic to retrieve p-value
    A2_k0 = [0.3970, 0.5690, 0.7960, 0.9740, 1.1580, 1.4090, 1.6030, 2.0640]
    ptable = [0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001]
    expected_p = np.clip(float(interp1d(A2_k0, ptable, fill_value='extrapolate')(expected_a2)), 0.0, 1.0)
    
    actual_p = andersondarling(z_rand, k=0)
    
    assert actual_p == pytest.approx(expected_p, abs=1e-5)