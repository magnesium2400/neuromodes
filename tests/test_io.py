import os
from pathlib import Path
from tempfile import TemporaryDirectory
from joblib import Memory
from lapy import TriaMesh
import numpy as np
from pytest import raises
from neuromodes.io import read_surf, fetch_surf, fetch_map, _set_cache
from neuromodes.mesh import check_surf

def test_fetch_surf():
    for hemi in ['L', 'R']:
        for species in ['human', 'macaque', 'marmoset']:
            for density in ['4k', '32k']:
                if species != 'human' and density == '4k':
                    with raises(ValueError, match="Surface data .* not found"):
                        fetch_surf(species=species, hemi=hemi, density=density)
                    continue

                surf, medmask = fetch_surf(species=species, hemi=hemi, density=density)
                assert surf.v.shape[0] > 0
                assert surf.v.shape[1] == 3
                assert surf.t.shape[0] > 0
                assert surf.t.shape[1] == 3
                assert medmask.dtype == bool
                assert medmask.shape == (surf.v.shape[0],)

                check_surf(surf)  # Should not raise

def test_fetch_invalid_surf():
    with raises(ValueError, match="Surface data .* not found"):
        fetch_surf(surf_type='makessense')

def test_fetch_gradient():
    grad = fetch_map('fcgradient1')
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (32492,)

def test_fetch_invalid_map():
    with raises(ValueError, match="Map 'sp-human_tpl-fsLR_den-32k_hemi-L_panshifu.func.gii'.*"):
        fetch_map('panshifu')

def test_read_surf_dict():
    surf_data = {
        'vertices': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        'faces': [[0, 1, 2], [1, 2, 3]]
    }
    surf = read_surf(surf_data)
    assert isinstance(surf, TriaMesh)
    assert surf.v.shape == (4, 3)
    assert surf.t.shape == (2, 3)

def test_read_surf_vtk():
    vtk_surf = read_surf(
        Path(__file__).parent / 'test_data' / 'sp-human_tpl-fsaverage5_den-10k_hemi-L_midthickness.vtk'
        )

    assert isinstance(vtk_surf, TriaMesh)
    assert vtk_surf.v.shape == (10242, 3)
    assert vtk_surf.t.shape == (20480, 3)

def test_read_surf_invalid():
    invalid_path = Path(__file__).parent / 'test_data' / 'civilised_lunch.surf.vtk'
    with raises(ValueError, match="File not found: .*civilised_lunch.surf.vtk"):
        read_surf(invalid_path)

def test_read_surf_freesurfer():
    for surf_type in ['inflated', 'orig', 'pial', 'smoothwm', 'sphere', 'white']:
        fs_surf = read_surf(
            Path(__file__).parent / 'test_data' / f'fsaverage-lh.{surf_type}'
            )
         
        assert isinstance(fs_surf, TriaMesh)
        assert fs_surf.v.shape[0] > 100
        assert fs_surf.t.shape[0] > 100
        assert fs_surf.v.shape[1] == 3
        assert fs_surf.t.shape[1] == 3

# TODO: also test dict reading by just converting other formats to dict
def test_mesh_dict():
    # Surface case
    surf = {
        'vertices': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        'faces': [[0, 1, 2], [0, 2, 3]]
    }

    surf_triamesh = read_surf(surf)
    assert surf_triamesh.t.shape == (2, 3), \
        "read_surf should return a TriaMesh with the correct triangular connectivity"
    check_surf(surf_triamesh)

def test_caching():
    # Get CACHE_DIR
    cache_dir = os.getenv("CACHE_DIR")

    # Test with temporary directory
    with TemporaryDirectory() as temp_cache_dir:
        os.environ["CACHE_DIR"] = temp_cache_dir
        
        memory = _set_cache()
        assert isinstance(memory, Memory)
        assert str(memory.location) == temp_cache_dir
    
    # Restore original CACHE_DIR
    if cache_dir is not None:
        os.environ["CACHE_DIR"] = cache_dir
    elif "CACHE_DIR" in os.environ:
        del os.environ["CACHE_DIR"]

def test_caching_default_dir(capsys):
    # Temporarily unset CACHE_DIR
    cache_dir = os.getenv("CACHE_DIR")
    if "CACHE_DIR" in os.environ:
        del os.environ["CACHE_DIR"]

    try:
        # Invoke _set_cache and check default directory
        memory = _set_cache()
        expected_dir = Path.home() / ".neuromodes_cache"

        assert isinstance(memory, Memory)
        assert memory.location == expected_dir

        print_log = capsys.readouterr().out
        assert f"Using default cache directory at {expected_dir}" in print_log
    finally:
        # Restore original CACHE_DIR
        if cache_dir is not None:
            os.environ["CACHE_DIR"] = cache_dir