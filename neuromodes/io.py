"""
Module for loading surface meshes and maps, as well as setting up caching.
"""

from __future__ import annotations
from importlib.resources import files, as_file
from os import getenv
from pathlib import Path
from typing import Union, Tuple, cast, TYPE_CHECKING
from joblib import Memory
from lapy import TriaMesh
from nibabel.gifti.gifti import GiftiImage
from nibabel.loadsave import load

if TYPE_CHECKING:
    from numpy.typing import NDArray

fs_extensions = ('.white', '.pial', '.inflated', '.orig', '.sphere', '.smoothwm', '.qsphere',
                 '.fsaverage')

def read_surf(
    surf: Union[str, Path, GiftiImage, TriaMesh, dict]
) -> TriaMesh:
    """Load a triangular surface mesh.

    Parameters
    ----------
    surf : str, Path, GiftiImage, lapy.TriaMesh, or dict
        Surface mesh specified as a file path (string or Path) to a VTK (.vtk), GIFTI (.gii), or
        FreeSurfer file (.white, .pial, .inflated, .orig, .sphere, .smoothwm, .qsphere, .fsaverage),
        an instance of `nibabel.GiftiImage` or `lapy.TriaMesh`, or a dictionary
        with `'vertices'` and `'faces'` keys, referencing arrays of shapes (n_verts, 3) and
        (n_trias, 3), respectively.

    Returns
    -------
    lapy.TriaMesh
        Surface mesh with vertices and faces.

    Raises
    ------
    ValueError
        If `surf` is a path-like string to an unsupported format.
    ValueError
        If `surf` is a path-like string to a file that does not exist.
    """
    if isinstance(surf, TriaMesh):
        return surf
    elif isinstance(surf, GiftiImage):
        vertices=surf.darrays[0].data
        faces=surf.darrays[1].data
    elif isinstance(surf, dict):
        vertices=surf['vertices']
        faces=surf['faces']
    else:
        surf_str = str(surf)
        # check that file exists
        if not Path(surf_str).is_file():
            raise ValueError(f'File not found: {surf_str}')
        # Handle different file types
        if surf_str.endswith('.vtk'):
            return TriaMesh.read_vtk(surf_str)
        elif surf_str.endswith(fs_extensions):
            return TriaMesh.read_fssurf(surf_str)
        elif surf_str.endswith('.gii'):
            surf_data = cast(GiftiImage, load(surf_str)).darrays
            vertices=surf_data[0].data
            faces=surf_data[1].data
        else:
            raise ValueError(
                '`surf` must be a path-like string to a valid VTK (.vtk), GIFTI (.gii), or '
                f'FreeSurfer file {fs_extensions}, an instance of `nibabel.GiftiImage` or '
                '`lapy.TriaMesh`, or a dictionary of `faces` and `vertices` with shapes '
                '(n_verts, 3) and (n_trias, 3), respectively.'
                )
        
    return TriaMesh(v=vertices, t=faces)

def fetch_surf(
    species: str = 'human',
    density: str = '32k',
    hemi: str = 'L',
    surf_type: str = 'midthickness',
    template: str = 'fsLR'
) -> Tuple[TriaMesh, NDArray]:
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
    surf : lapy.TriaMesh
        The loaded surface mesh.
    medmask : np.ndarray
        The medial wall mask as a boolean array.

    Raises
    ------
    ValueError
        If the specified surface data is not found in the `neuromodes/data` directory.
    """
    data_dir = files('neuromodes.data')
    surf_name = f'sp-{species}_tpl-{template}_den-{density}_hemi-{hemi}_{surf_type}.surf.gii'
    mask_name = f'sp-{species}_tpl-{template}_den-{density}_hemi-{hemi}_medmask.label.gii'

    try:
        with as_file(data_dir / surf_name) as fpath:
            surf = read_surf(fpath)
        with as_file(data_dir / mask_name) as fpath:
            medmask = cast(GiftiImage, load(fpath)).darrays[0].data.astype(bool)
        
        return surf, medmask
    except Exception:
        raise ValueError(
            f"Surface data {surf_name} not found. Please see {data_dir}/included_data.csv or "
            "https://github.com/NSBLab/neuromodes/tree/main/neuromodes/data/included_data.csv for a"
            " list of available surfaces."
            )

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
    
    except Exception:
        raise ValueError(
            f"Map '{filename}' not found. Please see {data_dir}/included_data.csv or "
            "https://github.com/NSBLab/neuromodes/tree/main/neuromodes/data/included_data.csv for a"
            " list of available data files."
        )

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
    CACHE_DIR = getenv("CACHE_DIR")
    if CACHE_DIR is None:
        CACHE_DIR = Path.home() / ".neuromodes_cache"
        print(f"Using default cache directory at {CACHE_DIR}. To cache elsewhere, set the CACHE_DIR"
              " environment variable beforehand.")
    else:
        CACHE_DIR = Path(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    return Memory(CACHE_DIR, verbose=0)