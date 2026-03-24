import pytest
import csv
from pathlib import Path
import numpy as np
from lapy import TriaMesh

from neuromodes.eigen import EigenSolver
from neuromodes.mesh import estimate_fwhm, mode_to_group, group_to_mode, truncate_emodes
from neuromodes.io import fetch_surf, fetch_map

def test_fwhm_wb():
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

@pytest.mark.parametrize("method_and_rtol", [('wb', 1e-1), ('fem', 1e-6)])
def test_fwhm(method_and_rtol):
    mesh, _ = fetch_surf(surf_type='sphere', density='4k')
    solver = EigenSolver(mesh).solve(n_modes=25)
    estimated_value = estimate_fwhm(solver.geometry, solver.emodes[:,1:], method=method_and_rtol[0])
    theoretical_value = np.sqrt(8 * np.log(2) / solver.evals[1:])
    assert np.allclose(estimated_value, theoretical_value, rtol=method_and_rtol[1]), \
        f"Estimated value does not match theoretical value for method {method_and_rtol[0]}"

def test_mode_to_group():
    idx = np.arange(22)

    assert np.all(mode_to_group(idx, method='ceil') 
        == [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4])
    assert np.all(mode_to_group(idx, method='floor') 
        == [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3])
    assert np.all(mode_to_group(idx, method='round') 
        == [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4])
    
    assert np.allclose(mode_to_group(idx, method='raw'), 
        np.sqrt(np.arange(1,23)) - 1)

def test_group_to_mode():
    idx = mode_to_group(np.arange(22), method='raw')

    assert np.all(group_to_mode(idx, method='ceil') 
        == [0, 3, 3, 3, 8, 8, 8, 8, 8, 15, 15, 15, 15, 15, 15, 15, 24, 24, 24, 24, 24, 24])
    assert np.all(group_to_mode(idx, method='floor') 
        == [0, 0, 0, 3, 3, 3, 3, 3, 8,  8,  8,  8,  8,  8,  8, 15, 15, 15, 15, 15, 15, 15])
    assert np.all(group_to_mode(idx, method='round') 
        == [0, 0, 3, 3, 3, 3, 8, 8, 8,  8,  8,  8, 15, 15, 15, 15, 15, 15, 15, 15, 24, 24])
    
    assert np.allclose(group_to_mode(idx, method='raw'), 
        np.arange(22))

def test_mode_to_mode():
    """Tests that the 'raw' methods are perfectly invertible bijections."""
    # Mode -> Group -> Mode
    a = np.arange(0, 100)
    b = group_to_mode(mode_to_group(a, method='raw'), method='raw')
    np.testing.assert_array_almost_equal(a, b)

def test_group_to_group():
    """Tests that the 'raw' methods are perfectly invertible bijections."""
    # Group -> Mode -> Group
    a = np.linspace(0, 20, 100)
    b = mode_to_group(group_to_mode(a, method='raw'), method='raw')
    np.testing.assert_array_almost_equal(a, b)

@pytest.fixture(scope="module")
def solver(): 
    mesh, _ = fetch_surf(surf_type='sphere', density='4k')
    return EigenSolver(mesh).solve(n_modes=25)

def test_truncate_emodes(solver):
    betas = np.random.randn(solver.n_modes, 1)
    for n_zeros in range(1, solver.n_modes-2):
        # Create data using only part of the mode basis
        vfunc = solver.emodes @ np.vstack((betas[:-n_zeros], np.zeros((n_zeros,1))))
        expected_groups = np.ceil(np.sqrt(solver.n_modes - n_zeros)).astype(int)
        
        # Expected and actual outputs
        out = truncate_emodes(geometry=solver.geometry, vfunc=vfunc, 
            emodes=solver.emodes, evals=solver.evals, mass=solver.mass, power_threshold=1e-8)
        assert out == expected_groups, \
            f"Expected {expected_groups} groups, but got {out} groups when truncating {n_zeros} modes."
