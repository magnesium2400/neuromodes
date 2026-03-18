import pytest
import csv
from pathlib import Path
import numpy as np
from lapy import TriaMesh

from neuromodes.mesh import estimate_fwhm
from neuromodes.io import fetch_surf, fetch_map

def test_csv_reading():
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
