"""
Module for reading, validating, and manipulating surface meshes.
"""

from __future__ import annotations
from importlib.resources import files, as_file
import os
from pathlib import Path
from typing import Union, Tuple, cast, TYPE_CHECKING
from lapy import TriaMesh
from nibabel.freesurfer.io import read_geometry
from nibabel.gifti.gifti import GiftiImage
from nibabel.loadsave import load
import numpy as np
from trimesh import Trimesh

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike

def read_surf(
    surf: Union[str, Path, Trimesh, TriaMesh, dict]
) -> Trimesh:
    """Load and validate a surface mesh.

    Parameters
    ----------
    surf : str, Path, trimesh.Trimesh, lapy.TriaMesh, or dict
        Surface mesh specified as a file path (string or Path) to a VTK (.vtk), GIFTI (.gii), or
        FreeSurfer file (.white, .pial, .inflated, .orig, .sphere, .smoothwm, .qsphere, .fsaverage),
        an instance of `trimesh.Trimesh` or `lapy.TriaMesh`, or a dictionary with `vertices` and
        `faces` keys.

    Returns
    -------
    trimesh.Trimesh
        Validated surface mesh with vertices and faces.

    Raises
    ------
    ValueError
        If `surf` is a path-like string to an unsupported format.
    ValueError
        If `surf` is a path-like string to a file that does not exist.
    """
    if isinstance(surf, Trimesh):
        trimesh = surf
    elif isinstance(surf, TriaMesh):
        trimesh = Trimesh(vertices=surf.v, faces=surf.t)
    elif isinstance(surf, dict):
        trimesh = Trimesh(vertices=surf['vertices'], faces=surf['faces'])
    else:
        surf_str = str(surf)
        # check that file exists
        if not Path(surf_str).is_file():
            raise ValueError(f'File not found: {surf_str}')

        # Handle different file types
        if surf_str.endswith('.vtk'):
            surf_lapy = TriaMesh.read_vtk(surf_str)
            trimesh = Trimesh(vertices=surf_lapy.v, faces=surf_lapy.t)
        elif surf_str.endswith('.gii'):
            surf_data = cast(GiftiImage, load(surf_str)).darrays
            trimesh = Trimesh(vertices=surf_data[0].data, faces=surf_data[1].data)
        elif surf_str.endswith(
            ('white', 'pial', 'inflated', 'orig', 'sphere', 'smoothwm', 'qsphere', 'fsaverage')
            ):
            vertices, faces = read_geometry(
                surf_str, read_metadata=False, read_stamp=False
                ) # will only return two outputs now # type: ignore
            trimesh = Trimesh(vertices=vertices, faces=faces) # type: ignore
        else:
            raise ValueError(
                '`surf` must be a path-like string to a valid VTK (.vtk), GIFTI (.gii), or '
                'FreeSurfer file (.white, .pial, .inflated, .orig, .sphere, .smoothwm, .qsphere, '
                '.fsaverage), an instance of `trimesh.Trimesh` or `lapy.TriaMesh`, or a dictionary '
                'of `faces` and `vertices`.'
                )

    return trimesh

def mask_surf(
    surf: Trimesh,
    mask: ArrayLike
) -> Trimesh:
    """
    Remove specified vertices and corresponding faces from the surface mesh. Returns a validated 
    `trimesh.Trimesh` object.

    Parameters
    ----------
    surf : trimesh.Trimesh
        The input surface mesh.
    mask : array-like
        A boolean array indicating which vertices to keep (`True`) or remove (`False`).

    Returns
    -------
    trimesh.Trimesh
        The masked surface mesh.

    Raises
    ------
    ValueError
        If `mask` does not have a length matching the number of vertices in `surf`.

    Notes
    -----
    In `Trimesh.submesh`, `repair=False` is used to avoid an unnecessary dependency on 
    `networkx`. Mesh validation is handled separately in `check_surf` in `EigenSolver`.
    """
    mask = np.asarray(mask, dtype=bool)

    if mask.shape != (surf.vertices.shape[0],):
        raise ValueError(f"`mask` must have shape {(surf.vertices.shape[0],)} to match the number "
                         "of vertices in the surface mesh.")
    
    # Remove faces where all vertices are in the mask
    face_mask = np.all(mask[surf.faces], axis=1)
    masked_surf = surf.submesh([face_mask], repair=False)[0] #type: ignore # submesh returns a list by default
    
    return masked_surf

def check_surf(
    surf: Trimesh
) -> None:
    """
    Check if the surface mesh is contiguous with no unreferenced vertices.
    
    Parameters
    ----------
    surf : trimesh.Trimesh
        The surface mesh to check.

    Raises
    ------
    ValueError
        If the surface mesh contains unreferenced vertices.
    ValueError
        If the surface mesh is not contiguous.
    """
    # Check for unreferenced vertices
    referenced = np.zeros(len(surf.vertices), dtype=bool)
    referenced[surf.faces] = True
    if not np.all(referenced):
        raise ValueError(f'Surface mesh contains {np.sum(~referenced)} unreferenced '
                         'vertices (i.e., not part of any face).')

    # Check if the surf is contiguous
    n_components = surf.body_count
    if n_components != 1:
        raise ValueError(f'Surface mesh is not contiguous: {n_components} connected components '
                         'found.')

def fetch_surf(
    species: str = 'human',
    density: str = '32k',
    hemi: str = 'L',
    surf_type: str = 'midthickness',
    template: str = 'fsLR'
) -> Tuple[Trimesh, NDArray]:
    """
    Load a cortical triangular surface mesh and medial wall mask from neuromodes data directory. For
    a list of available surfaces, see
    https://github.com/NSBLab/neuromodes/tree/main/neuromodes/data/included_data.csv.

    Parameters
    ----------
    species : str, optional
        Species of the surface mesh. Options include `'human'`, `'macaque'`, and `'marmoset'`.
        Default is `'human'`.
    density : str, optional
        Density of the surface mesh. Options include `'32k'` for all species, and `'4k'` for human.
        Default is `'32k'`.
    hemi : str, optional
        Hemisphere of the surface mesh. Options are `'L'` for all species, and `'R'` for human.
        Default is `'L'`.
    surf_type : str, optional
        Surface type to load. Currently only supports `'midthickness'`. Default is `'midthickness'`.
    template : str, optional
        Template of the surface mesh. Currently only supports `'fsLR'`. Default is `'fsLR'`.
    
    Returns
    -------
    surf : trimesh.Trimesh
        The loaded surface mesh.
    medmask : np.ndarray
        The medial wall mask as a boolean array.

    Raises
    ------
    ValueError
        If the specified surface data is not found in the `neuromodes/data` directory.
    """
    data_dir = files('neuromodes.data')
    surfname = f'sp-{species}_tpl-{template}_den-{density}_hemi-{hemi}_{surf_type}.surf.gii'
    maskname = f'sp-{species}_tpl-{template}_den-{density}_hemi-{hemi}_medmask.label.gii'

    try:
        with as_file(data_dir / surfname) as fpath:
            surf = read_surf(fpath)
        with as_file(data_dir / maskname) as fpath:
            medmask = cast(GiftiImage, load(fpath)).darrays[0].data.astype(bool)
        
        return surf, medmask
    except Exception as e:
        raise ValueError(
            f"Surface data not found. Please see {data_dir}/included_data.csv or "
            "https://github.com/NSBLab/neuromodes/tree/main/neuromodes/data/included_data.csv for a"
            " list of available surfaces."
            ) from e

def fetch_map(
    data: str,
    species: str = 'human',
    density: str = '32k',
    hemi: str = 'L',
    template: str = 'fsLR'
) -> NDArray:
    """
    Load cortical surface data from neuromodes data directory. For a list of available maps, see
    https://github.com/NSBLab/neuromodes/tree/main/neuromodes/data/included_data.csv.

    Parameters
    ----------
    data : str
        Cortical map to load. Options include `'fcgradient1'`, `'myelinmap'`, `'ndi'`, `'odi'`, and
        `'thickness'`.
    species : str, optional
        Species of the surface mesh. Currently only supports `'human'`. Default is `'human'`.
    density : str, optional
        Density of the surface mesh. Currently only supports `'32k'`. Default is `'32k'`.
    hemi : str, optional
        Hemisphere of the surface mesh. Currently only supports `'L'`. Default is `'L'`.
    template : str, optional
        Template of the surface mesh. Currently only supports `'fsLR'`. Default is `'fsLR'`.

    Returns
    -------
    np.ndarray
        The loaded cortical map data.

    Raises
    ------
    ValueError
        If the specified map data is not found in the `neuromodes/data` directory.
    """
    data_dir = files('neuromodes.data')
    filename = f'sp-{species}_tpl-{template}_den-{density}_hemi-{hemi}_{data}.func.gii'

    try:
        with as_file(data_dir / filename) as fpath:
            return cast(GiftiImage, load(fpath)).darrays[0].data
    
    except Exception as e:
        raise ValueError(
            f"Map '{filename}' not found. Please see {data_dir}/included_data.csv or "
            "https://github.com/NSBLab/neuromodes/tree/main/neuromodes/data/included_data.csv for a"
            " list of available data files."
        ) from e

def _set_cache():
    """
    Set up joblib memory caching based. Uses the directory specified by the `CACHE_DIR` 
    environment variable, or defaults to `~/.neuromodes_cache` if not set.
    
    Returns
    -------
    joblib.Memory
        Configured joblib Memory object for caching.

    Raises
    ------
    ImportError
        If `joblib` is not installed.
    """
    try:
        from joblib import Memory
    except ImportError:
        raise ImportError(
            "joblib is required for caching. Please install it via 'pip install joblib' (or "
            "install neuromodes with UV via 'uv add \"neuromodes[cache] @ "
            "git+https://github.com/NSBLab/neuromodes.git\")."
        )

    CACHE_DIR = os.getenv("CACHE_DIR")
    if CACHE_DIR is None:
        CACHE_DIR = Path.home() / ".neuromodes_cache"
        print(f"Using default cache directory at {CACHE_DIR}. To cache elsewhere, set the CACHE_DIR"
              " environment variable beforehand.")
    else:
        CACHE_DIR = Path(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    return Memory(CACHE_DIR, verbose=0)