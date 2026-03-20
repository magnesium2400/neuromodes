import pytest
import csv
from pathlib import Path
import numpy as np
from lapy import TriaMesh

from neuromodes.mesh import estimate_fwhm, mode_to_group, group_to_mode
from neuromodes.io import fetch_surf, fetch_map

def test_fwhm_regression():
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
        estimated_fwhm = estimate_fwhm(geometry, vfunc)

        assert np.isclose(estimated_fwhm, float(expected_fwhm), atol=0.01), \
            f"Estimated FWHM ({estimated_fwhm}) does not match expected FWHM ({float(expected_fwhm)}) for {data}"

def test_mode_to_group():
    idx = np.arange(22)

    assert np.all(mode_to_group(idx, method='ceil') \
        == [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4])
    assert np.all(mode_to_group(idx, method='floor') \
        == [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3])
    assert np.all(mode_to_group(idx, method='round') \
        == [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4])
    
    assert np.allclose(mode_to_group(idx, method='raw'), \
        np.sqrt(np.arange(1,23)) - 1)

def test_group_to_mode():
    idx = mode_to_group(np.arange(22), method='raw')

    assert np.all(group_to_mode(idx, method='ceil') \
        == [0, 3, 3, 3, 8, 8, 8, 8, 8, 15, 15, 15, 15, 15, 15, 15, 24, 24, 24, 24, 24, 24])
    assert np.all(group_to_mode(idx, method='floor') \
        == [0, 0, 0, 3, 3, 3, 3, 3, 8,  8,  8,  8,  8,  8,  8, 15, 15, 15, 15, 15, 15, 15])
    assert np.all(group_to_mode(idx, method='round') \
        == [0, 0, 3, 3, 3, 3, 8, 8, 8,  8,  8,  8, 15, 15, 15, 15, 15, 15, 15, 15, 24, 24])
    
    assert np.allclose(group_to_mode(idx, method='raw'), \
        np.arange(22))

def test_continuous_bijection():
    """Tests that the 'raw' methods are perfectly invertible bijections."""
    
    # Mode -> Group -> Mode
    a = np.arange(0, 100)
    b = group_to_mode(mode_to_group(a, method='raw'), method='raw')
    np.testing.assert_array_almost_equal(a, b)

    # Group -> Mode -> Group
    a = np.linspace(0, 20, 100)
    b = mode_to_group(group_to_mode(a, method='raw'), method='raw')
    np.testing.assert_array_almost_equal(a, b)
