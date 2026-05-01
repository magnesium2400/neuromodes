import pytest
import csv
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from lapy import TriaMesh

from neuromodes.eigen import EigenSolver
from neuromodes.mesh import (
    estimate_fwhm, mode_to_group, group_to_mode, truncate_emodes, convert_to_eigenvalue, convert_from_eigenvalue)
from neuromodes.io import fetch_surf, fetch_map

class TestEigenvalueConversion:
    # Tests with manually specified values
    def test_convert_to_eigenvalue_exact_values(self):
        """Tests that the mathematical formulas are applied correctly."""
        area = 4 * np.pi # sphere of radius 1
        assert_allclose(convert_to_eigenvalue(2.0, 'eigenvalue'),       2.0)
        assert_allclose(convert_to_eigenvalue(2.0, 'wavelength'),       np.pi**2)
        assert_allclose(convert_to_eigenvalue(2.0, 'fwhm'),             2 * np.log(2))
        assert_allclose(convert_to_eigenvalue(2.0, 'group', area=area), 6.0)
        assert_allclose(convert_to_eigenvalue(2.0, 'mode', area=area),  2.0)

    def test_convert_from_eigenvalue_exact_values(self):
        """Tests the inverse mathematical formulas."""
        area = 4 * np.pi # sphere of radius 1
        assert_allclose(convert_from_eigenvalue(2.0,       'eigenvalue'), 2.0)
        assert_allclose(convert_from_eigenvalue(np.pi**2,  'wavelength'), 2.0)
        assert_allclose(convert_from_eigenvalue(2 * np.log(2),   'fwhm'), 2.0)
        assert_allclose(convert_from_eigenvalue(6.0, 'group', area=area), 2.0)
        assert_allclose(convert_from_eigenvalue(2.0,  'mode', area=area), 2.0)

    # Test inversions
    @pytest.mark.parametrize('ctype', ['wavelength', 'fwhm', 'group', 'mode'])
    def test_inversions(self, ctype):
        """Tests that converting to and from an eigenvalue returns the original array."""
        area = 100.0
        values1 = np.array([0.5, 1.0, 2.5, 10.0, 100.0])
        evals = convert_to_eigenvalue(values1, input=ctype, area=area)
        values2 = convert_from_eigenvalue(evals, output=ctype, area=area)
        assert_allclose(values1, values2)

    # Exception and validation Tests
    @pytest.mark.parametrize('direction', [convert_to_eigenvalue, convert_from_eigenvalue])
    @pytest.mark.parametrize('ctype', ['group', 'mode'])
    def test_missing_area_errors(self, direction, ctype):
        """Tests that missing area arguments trigger ValueErrors."""
        with pytest.raises(ValueError, match="Area must be provided"):
            direction(5.0, ctype) 

    @pytest.mark.parametrize('direction', [convert_to_eigenvalue, convert_from_eigenvalue])
    def test_invalid_type_errors(self, direction):
        """Tests that unrecognized string types trigger ValueErrors."""
        with pytest.raises(ValueError, match=r"Incorrect .*put specified"):
            direction(5.0, 'frequency')

    # Zero and inf
    @pytest.mark.parametrize('direction', [convert_to_eigenvalue, convert_from_eigenvalue])
    @pytest.mark.parametrize('ctype', ['wavelength', 'fwhm'])
    def test_zero_division_handling(self, direction, ctype):
        """Tests that the context managers successfully handle division by zero."""
        with pytest.warns(RuntimeWarning, match="divide by zero encountered in .*"):
            # direction(0.0, ctype)
            evals = np.array([0.0, 1.0, 1.0, 2.0])
            out = direction(evals, ctype)
            assert np.isinf(out[0])
            assert not np.isinf(out[1:]).any()

class TestModeGroupConversion:
    # First test perfect squares
    @pytest.mark.parametrize("method", ['ceil', 'floor', 'round', 'raw'])
    def test_mode_to_group_squares(self, method): 
        n = np.arange(1,22)**2-1
        assert np.all(mode_to_group(n, method=method) == np.sqrt(n+1)-1), \
            f"Failed for method {method} with perfect squares."

    # Then test non-perfect squares with manually calculated expected values
    def test_mode_to_group(self):
        idx = np.arange(22)
        assert np.all(mode_to_group(idx, method='ceil') 
            == [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4])
        assert np.all(mode_to_group(idx, method='floor') 
            == [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3])
        assert np.all(mode_to_group(idx, method='round') 
            == [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4])
        assert np.allclose(mode_to_group(idx, method='raw'), 
            np.sqrt(np.arange(1,23)) - 1)

    # First test perfect squares
    @pytest.mark.parametrize("method", ['ceil', 'floor', 'round', 'raw'])
    def test_group_to_mode_squares(self, method): 
        n = np.arange(22)
        assert np.all(group_to_mode(n, method=method) == (n + 1)**2 - 1), \
            f"Failed for method {method} with perfect squares."

    # Then test non-perfect squares with manually calculated expected values
    def test_group_to_mode(self):
        idx = mode_to_group(np.arange(22), method='raw')
        assert np.all(group_to_mode(idx, method='ceil') 
            == [0, 3, 3, 3, 8, 8, 8, 8, 8, 15, 15, 15, 15, 15, 15, 15, 24, 24, 24, 24, 24, 24])
        assert np.all(group_to_mode(idx, method='floor') 
            == [0, 0, 0, 3, 3, 3, 3, 3, 8,  8,  8,  8,  8,  8,  8, 15, 15, 15, 15, 15, 15, 15])
        assert np.all(group_to_mode(idx, method='round') 
            == [0, 0, 3, 3, 3, 3, 8, 8, 8,  8,  8,  8, 15, 15, 15, 15, 15, 15, 15, 15, 24, 24])
        assert np.allclose(group_to_mode(idx, method='raw'), 
            np.arange(22))

    # Mode -> Group -> Mode
    def test_mode_to_mode(self):
        """Tests that the 'raw' methods are perfectly invertible bijections."""
        a = np.arange(0, 100)
        b = group_to_mode(mode_to_group(a, method='raw'), method='raw')
        np.testing.assert_array_almost_equal(a, b)

    # Group -> Mode -> Group
    def test_group_to_group(self):
        """Tests that the 'raw' methods are perfectly invertible bijections."""
        a = np.linspace(0, 20, 100)
        with pytest.warns(UserWarning, match="mode_id should be an integer or array of integers."):
            b = mode_to_group(group_to_mode(a, method='raw'), method='raw')
        np.testing.assert_array_almost_equal(a, b)

@pytest.fixture(scope="module")
def solver(): 
    mesh, _ = fetch_surf(surf_type='sphere', density='4k')
    return EigenSolver(mesh).solve(n_modes=100)

class TestFWHM:
    # Test that (i) both methods run on 2d data; (ii) 'fem' is very close to theory; (iii) 'wb'
    # approximately equals theory
    @pytest.mark.parametrize("method_and_rtol", [('wb', 1e-1), ('fem', 1e-6)])
    def test_fwhm(self, solver, method_and_rtol):
        estimated_value = estimate_fwhm(solver.geometry, solver.emodes[:,1:], method=method_and_rtol[0])
        theoretical_value = np.sqrt(8 * np.log(2) / solver.evals[1:])
        assert np.allclose(estimated_value, theoretical_value, rtol=method_and_rtol[1]), \
            f"Estimated value does not match theoretical value for method {method_and_rtol[0]}"

    # Test that 'wb' is very close to values previously computed using wb_command
    def test_fwhm_wb(self):
        ATOL = 1e-6
        # Do some CSV parsing without using pandas
        filename = Path(__file__).parent / 'test_data' / 'mesh_estimate_fwhm_results.csv'
        with open(filename, mode='r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            data_list = list(csv_reader)
        header = data_list[0]  # first row is the header

        # Compare with saved values
        for row in data_list[1:]:
            species = row[header.index('species')]
            template = row[header.index('template')]
            density = row[header.index('density')]
            hemi = row[header.index('hemi')]
            surf_type = row[header.index('surf_type')]
            data = row[header.index('data')]
            expected_fwhm = row[header.index('wbcommand_fwhm')]

            mesh, _ = fetch_surf(surf_type=surf_type, template=template, species=species, density=density, hemi=hemi)
            geometry = TriaMesh(mesh.vertices, mesh.faces)
            vfunc = fetch_map(data=data, template=template, species=species, density=density, hemi=hemi)
            estimated_fwhm = estimate_fwhm(geometry, vfunc, method='wb')

            assert np.allclose(estimated_fwhm, float(expected_fwhm), atol=ATOL), \
                f"Estimated FWHM ({estimated_fwhm}) does not match expected FWHM ({float(expected_fwhm)}) for {data}"

    # Test that roi_mask runs without errors and produces different results than global FWHM (but
    # doesn't test actual correctness)
    # TODO : add correctness tests for method fem (what is ground truth)? 
    @pytest.mark.parametrize("method", ['wb', 'fem'])
    def test_roi_mask_execution(self, solver, method):
        """Tests that the roi_mask successfully evaluates without shape errors."""
        # Create a dummy mask (e.g., just the first half of the vertices)
        num_vertices = solver.geometry.v.shape[0]
        mask = np.zeros(num_vertices, dtype=bool)
        mask[:num_vertices // 2] = True
        
        # Test 1D array
        vfunc_1d = solver.emodes[:, 1]
        fwhm_1d = estimate_fwhm(solver, vfunc_1d, method=method, roi_mask=mask)
        assert isinstance(fwhm_1d, float)
        assert not np.isnan(fwhm_1d)
        
        # Verify FWHM is actually different than the global FWHM
        global_fwhm = estimate_fwhm(solver, vfunc_1d, method=method, roi_mask=None)
        assert fwhm_1d != global_fwhm

        # Test 2D array
        vfunc_2d = solver.emodes[:, 1:]
        fwhm_2d = estimate_fwhm(solver, vfunc_2d, method=method, roi_mask=mask)
        assert isinstance(fwhm_2d, np.ndarray)
        assert fwhm_2d.shape == (solver.n_modes - 1,)
        assert not np.isnan(fwhm_2d).any()
        assert not np.isinf(fwhm_2d).any()

class TestTruncateEmodes:
    # Test correctness of decompose
    def test_truncate_emodes_decompose(self, solver):
        betas = np.random.default_rng().standard_normal((solver.n_modes, 1))
        betas[0] = 0
        vfunc = solver.emodes @ betas 

        power = np.cumsum(betas**2)
        power = power / power[-1]
        
        kmin = 3
        thresholds = 1 - (power[kmin-1:-1] + power[kmin:]) / 2

        for i in range(kmin, solver.n_modes - 1):
            out = truncate_emodes(
                geometry=solver.geometry, 
                vfunc=vfunc, 
                emodes=solver.emodes, 
                mass=solver.mass, 
                threshold_method='decompose', 
                threshold=thresholds[i - kmin], 
                output='mode'
            )
            assert out == i + 1, f"Expected {i + 1} modes, but got {out}."

    # Test correctness of reconstruct
    def test_truncate_emodes_reconstruct(self, solver):
        betas = np.random.default_rng().standard_normal((solver.n_modes, 1))
        betas[0] = 0
        vfunc = solver.emodes @ betas 

        # TODO: change to new reconstruct
        _, errors, _ = solver.reconstruct(data=vfunc)

        kmin = 3
        thresholds = (errors[kmin-1:-1] + errors[kmin:]) / 2

        for i in range(kmin, solver.n_modes - 1):
            out = truncate_emodes(
                geometry=solver.geometry, 
                vfunc=vfunc, 
                emodes=solver.emodes, 
                mass=solver.mass, 
                threshold_method='reconstruct', 
                threshold=thresholds[i - kmin], 
                output='mode'
            )
            assert out == i + 1, f"Expected {i + 1} modes, but got {out}."
    
    # Test correctness of physical methods
    @pytest.mark.parametrize("threshold_method", ['eigenvalue', 'fwhm', 'wavelength'])
    def test_truncate_emodes_physical(self, solver, threshold_method):
        kmin = 3
        thresholds = (solver.evals[kmin-1:-1] + solver.evals[kmin:]) / 2
        thresholds = convert_from_eigenvalue(thresholds, output=threshold_method)

        for i in range(kmin, solver.n_modes - 1):
            out = truncate_emodes(
                geometry=solver.geometry, 
                vfunc=solver.emodes[:, i],
                evals=solver.evals, 
                threshold_method=threshold_method, 
                threshold=thresholds[i - kmin], 
                output='mode'
            )
            assert out == i, f"Expected {i} modes, but got {out}."

    # Test auto-thresholding for physical methods
    @pytest.mark.parametrize("threshold_method", ['eigenvalue', 'fwhm', 'wavelength'])
    def test_truncate_emodes_auto_threshold(self, solver, threshold_method):
        """Tests that passing threshold=None successfully triggers the FWHM estimator."""
        idx = np.random.default_rng().integers(1, solver.n_modes-1)
        vfunc = solver.emodes[:, idx] # Use a specific mode as the map
        
        # It should run without raising errors
        out_mode = truncate_emodes(
            geometry=solver.geometry, 
            vfunc=vfunc, 
            threshold=None, # Trigger auto-fallback
            evals=solver.evals, 
            threshold_method=threshold_method, 
            output='mode'
        )
        assert isinstance(out_mode, (int, np.integer))
        assert 0 < out_mode <= idx + 1

    # Test correctness of statistical thresholding using maps with some betas=0 
    @pytest.mark.parametrize("threshold_method", ['decompose', 'reconstruct'])
    def test_truncate_emodes_2d_statistical(self, solver, threshold_method):
        betas = np.random.default_rng().integers(1, 2, size=(solver.n_modes, solver.n_modes-3))
        filt = 1 - np.tri(*betas.shape, k=-3).astype(int) 
        vfunc = solver.emodes @ (betas * filt)
        expected_modes = filt.sum(axis=0)
        out = truncate_emodes(geometry=solver.geometry, vfunc=vfunc, 
            emodes=solver.emodes, evals=solver.evals, mass=solver.mass, 
            threshold_method=threshold_method, threshold=1e-8, output='mode')
        assert np.array_equal(out, expected_modes)

    # Test correctness of physical thresholding using maps with some betas=0
    @pytest.mark.parametrize("threshold_method", ['eigenvalue', 'fwhm', 'wavelength'])
    def test_truncate_emodes_2d_physical(self, solver, threshold_method):
        betas = np.random.default_rng().integers(1, 2, size=(solver.n_modes, solver.n_modes-3))
        filt = 1 - np.tri(*betas.shape, k=-3).astype(int) 
        vfunc = solver.emodes @ (betas * filt)
        thresholds = [solver.evals[i-1] for i in filt.sum(axis=0)]
        thresholds = convert_from_eigenvalue(thresholds, output=threshold_method)
        expected_modes = filt.sum(axis=0)
        out = truncate_emodes(geometry=solver.geometry, vfunc=vfunc, 
            emodes=solver.emodes, evals=solver.evals, mass=solver.mass, 
            threshold_method=threshold_method, threshold=thresholds, output='mode')
        assert np.array_equal(out, expected_modes)

    # Test some exceptions
    def test_truncate_emodes_exceptions(self, solver):
        """Tests that missing requirements trigger the correct errors."""
        vfunc = solver.emodes[:, 1]
        
        # Missing emodes/mass for decompose
        with pytest.raises(ValueError, match="emodes and mass must be provided"):
            truncate_emodes(geometry=solver.geometry, vfunc=vfunc, threshold=0.05, 
                            threshold_method='decompose')
            
        # Missing evals for physical properties
        with pytest.raises(ValueError, match="evals must be provided"):
            truncate_emodes(geometry=solver.geometry, vfunc=vfunc, threshold=10.0, 
                            threshold_method='wavelength')
            
        # Mismatched threshold array length
        vfunc_2d = solver.emodes[:, 1:3] # 2 maps
        with pytest.raises(ValueError, match="Length of threshold array must match"):
            truncate_emodes(geometry=solver.geometry, vfunc=vfunc_2d, threshold=[0.05], # Only 1 threshold
                            emodes=solver.emodes, mass=solver.mass)
            