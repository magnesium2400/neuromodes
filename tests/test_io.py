from tempfile import TemporaryDirectory
import os
from pathlib import Path
from unittest.mock import patch
from joblib import Memory
import numpy as np
from pytest import raises
from trimesh import Trimesh
from neuromodes.io import check_surf, fetch_surf, fetch_map, read_surf, _set_cache

def test_mesh_unreferenced_verts():
    # Create an invalid mesh with unreferenced vertices
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 2]])  # Last vertex unreferenced
    faces = np.array([[0, 1, 2], [0, 2, 3]])  # Only uses first 4 vertices, vertex 4 is unreferenced
    
    # IMPORTANT: Use process=False to prevent trimesh from automatically cleaning up unreferenced vertices    
    invalid_mesh = Trimesh(vertices=vertices, faces=faces, process=False)
   
    # check_surf should raise ValueError due to unreferenced vertex
    with raises(ValueError, match="Surface mesh contains .* unreferenced vertices"):
        check_surf(invalid_mesh)

def test_mesh_not_contiguous():
    # Create two separate triangles (disconnected components)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])  # Two separate triangles
    disconnected_mesh = Trimesh(vertices=vertices, faces=faces)
    
    # check_surf should raise ValueError due to multiple components
    with raises(ValueError, match="Surface mesh is not contiguous.*connected components"):
        check_surf(disconnected_mesh)

def test_fetch_surf():
    mesh, _ = fetch_surf()
    assert isinstance(mesh, Trimesh)
    assert mesh.vertices.shape == (32492, 3)

def test_fetch_medmask():
    _, medmask = fetch_surf(species='marmoset')
    assert isinstance(medmask, np.ndarray)
    assert medmask.dtype == bool
    assert medmask.shape == (32492,)
    assert np.sum(medmask) == 23052

def test_fetch_invalid_surf():
    with raises(ValueError, match="Surface data not found"):
        fetch_surf(surf_type='makessense')

def test_fetch_gradient():
    grad = fetch_map('fcgradient1')
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (32492,)

def test_fetch_invalid_map():
    with raises(ValueError, match="Map 'sp-human_tpl-fsLR_den-32k_hemi-L_panshifu.func.gii'.*"):
        fetch_map('panshifu')

def test_read_surf_dict():
    mesh_data = {
        'vertices': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        'faces': [[0, 1, 2], [1, 2, 3]]
    }
    mesh = read_surf(mesh_data)
    assert isinstance(mesh, Trimesh)
    assert mesh.vertices.shape == (4, 3)
    assert mesh.faces.shape == (2, 3)
    
    mesh_data_numpy = {
        'vertices': np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        'faces': np.array([[0, 1, 2], [1, 2, 3]])
    }
    mesh = read_surf(mesh_data_numpy)
    assert isinstance(mesh, Trimesh)
    assert mesh.vertices.shape == (4, 3)
    assert mesh.faces.shape == (2, 3)

    invalid_data = {
        'faces': np.array([[0, 1, 2]])
    }
    with raises(KeyError, match="'vertices'"):
        read_surf(invalid_data)

    invalid_data = {
        'vertices': np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    }
    with raises(KeyError, match="'faces'"):
        read_surf(invalid_data)

def test_read_surf_vtk():
    vtk_mesh = read_surf(
        Path(__file__).parent / 'test_data' / 'sp-human_tpl-fsaverage5_den-10k_hemi-L_midthickness.vtk'
        )

    assert isinstance(vtk_mesh, Trimesh)
    assert vtk_mesh.vertices.shape == (10242, 3)
    assert vtk_mesh.faces.shape == (20480, 3)

def test_read_surf_invalid():
    invalid_path = Path(__file__).parent / 'test_data' / 'civilised_lunch.surf.vtk'
    with raises(ValueError, match="File not found: .*civilised_lunch.surf.vtk"):
        read_surf(invalid_path)

def test_read_surf_freesurfer():
    for surf_type in ['inflated', 'orig', 'pial', 'smoothwm', 'sphere', 'white']:
        fs_mesh = read_surf(
            Path(__file__).parent / 'test_data' / f'fsaverage-lh.{surf_type}'
            )
         
        assert isinstance(fs_mesh, Trimesh)
        assert fs_mesh.vertices.shape[0] > 100
        assert fs_mesh.faces.shape[0] > 100
        assert fs_mesh.vertices.shape[1] == 3
        assert fs_mesh.faces.shape[1] == 3

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

def test_caching_no_joblib():
    # Mock the import of joblib to raise ImportError
    with patch.dict('sys.modules', {'joblib': None}):
        with raises(ImportError, match="joblib is required for caching"):
            _set_cache()