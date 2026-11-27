"""
Module for validating and manipulating meshes of brain structures.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable, TypeAlias, TypeVar
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from lapy import TetMesh, TriaMesh
from nibabel.affines import apply_affine
from nibabel.gifti.gifti import GiftiImage
from nibabel.nifti1 import Nifti1Image
from nibabel.nifti2 import Nifti2Image
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
from scipy.ndimage import binary_closing
from scipy.spatial import KDTree, Delaunay

from neuromodes.io import fs_extensions

_MeshT = TypeVar("_MeshT", TriaMesh, TetMesh)
_CallableFloatingToBool: TypeAlias = Callable[[NDArray[np.floating]], NDArray[np.bool_]]

def is_vol(
    geometry: TetMesh | TriaMesh | GiftiImage | str | Path | dict
) -> bool:
    """
    Determine whether the given geometry represents a volume or surface mesh.

    Parameters
    ----------
    geometry : lapy.TetMesh, lapy.TriaMesh, nibabel.gifti.GiftiImage, str, Path, or dict
        The geometry to check. Can be an instance of `lapy.TetMesh`, `lapy.TriaMesh`,
        `nibabel.gifti.GiftiImage`, a path-like string, or a dictionary with mesh data.

    Returns
    -------
    bool
        True if the geometry is a volume mesh, False if it is a surface mesh.

    Raises
    ------
    ValueError
        If the geometry is a path-like string with an unrecognized file extension.
    ValueError
        If the geometry is a dictionary that does not have keys 'vertices' and 'cells', or if
        'cells' does not reference an array with shape (n_tetras, 4) for volumes or (n_trias, 3) for
        surfaces.
    """
    # Instances
    if isinstance(geometry, TetMesh):
        return True
    if isinstance(geometry, (TriaMesh, GiftiImage)):
        return False
    
    # Paths
    if isinstance(geometry, (str, Path)):
        if str(geometry).endswith('.tetra.vtk'):
            return True
        elif str(geometry).endswith(('.vtk', '.gii') + fs_extensions):
            return False
        raise ValueError(
            'Received path-like string for `geometry`, but file extension is not recognized. '
            'Please provide a path-like string to a mesh file for a surface (.vtk, .gii, '
            f'{", ".join(fs_extensions)}) or volume (.tetra.vtk).')
    
    # Dictionary
    if isinstance(geometry, dict):
        err_str = ('Received an invalid dictionary for `geometry`. `vertices` key should reference '
                   'an array of shape (n_verts, 3) and `cells` key should reference an array of '
                   'shape (n_tetras, 4) for volumes or (n_trias, 3) for surfaces.')
        if 'vertices' not in geometry:
            raise ValueError(err_str)
        try:
            verts_per_face = np.asarray(geometry['cells']).shape[1]
        except (ValueError, KeyError, IndexError):
            raise ValueError(err_str)
        if verts_per_face == 4:
            return True
        elif verts_per_face == 3:
            return False
        raise ValueError(err_str)

def mask_mesh(
    geometry: _MeshT,
    mask: NDArray[np.bool_]
) -> _MeshT:
    """
    Remove specified vertices and corresponding cells from a triangular surface mesh. Note that this
    may produce a non-contiguous mesh with unreferenced vertices--use :func:`check_surf` to validate
    the resulting mesh.

    Parameters
    ----------
    geometry : lapy.TriaMesh
        The input surface mesh.
    mask : array-like
        A boolean array indicating which vertices to keep (``True``) or remove (``False``).

    Returns
    -------
    lapy.TriaMesh
        The masked surface mesh.

    Raises
    ------
    ValueError
        If ``mask`` does not have a length matching the number of vertices in ``geometry``.
    """
    # Format / validate arguments
    mask = np.asarray_chkfinite(mask, dtype=bool)
    if mask.shape != (geometry.v.shape[0],):
        raise ValueError(f"mask must have shape ({geometry.v.shape[0]},), matching the number of "
                         "vertices in geometry.")

    # Remove vertices not in mask
    v_masked = geometry.v[mask] # inherit original type

    # Update vertex indices of elements
    v_map = unmask_data(np.arange(np.sum(mask)), mask, fill_value=0).astype(geometry.t.dtype) 
    t_remapped = v_map[geometry.t]
    
    # Keep only elements where all vertices are in the mask
    elem_mask = np.all(mask[geometry.t], axis=1)
    t_masked = t_remapped[elem_mask]

    # Create a new TriaMesh with the masked vertices and elements
    return geometry.__class__(v=v_masked, t=t_masked)

def unmask_data(
    data: NDArray[np.floating],
    mask: NDArray[np.bool_],
    fill_value: float = np.nan
) -> NDArray[np.floating]:
    """
    Unmasks data by inserting it into a full array with the same length as the medial wall mask.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be unmasked, of shape ``(n_verts)`` or ``(n_verts, ...)``, where ``n_verts`` is
        the number of vertices in the masked mesh and additional axes represent maps to be unmasked
        independently.
    mask : numpy.ndarray
        A boolean array-like of shape ``(n_verts + n_extra_verts)``, where ``True`` indicates the
        positions of the data in the full array. Must contain exactly ``n_verts`` ``True`` values.
    fill_value : float, optional
        The value to fill in the positions outside the mask. Default is NaN.

    Returns
    -------
    numpy.ndarray
        The unmasked data of shape ``(n_verts + n_extra_verts)`` or
        ``(n_verts + n_added_verts, n_maps)``.

    Raises
    ------
    ValueError
        If ``mask`` is not a 1D boolean array.
    ValueError
        If ``data`` does not have shape ``(n_verts,)`` or ``(n_verts, n_maps)``.
    """
    # Format / validate arguments (TODO: use EigenData)
    mask = np.asarray_chkfinite(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("`mask` must be a 1D boolean array.")
    
    data = np.asarray(data)
    if data.shape[0] != np.sum(mask):
        raise ValueError("`data` must have shape (n_verts,...), where n_verts "
                         f"matches the number of True values in `mask` ({np.sum(mask)}).")
    
    # out_size
    out_size = (len(mask),) + data.shape[1:]

    # out_dtype: the safest dtype that can hold BOTH the data and the fill_value
    out_dtype = np.result_type(data.dtype, np.array(fill_value).dtype)
    
    # Initialise array of fill values
    data_unmasked = np.full(out_size, fill_value, dtype=out_dtype)

    # Overwrite rows with data where mask is True
    data_unmasked[mask, ...] = data

    return data_unmasked

def normalize_vol(
    geometry: TetMesh
) -> TetMesh:
    """
    Translate the mesh centroid to the origin and rescale to unit volume.

    Parameters
    ----------
    geometry : lapy.TetMesh
        The input volume mesh, to be modified in-place.
    """
    # Make copy to avoid in-place modification
    geometry = TetMesh(geometry.v, geometry.t)

    # Get edge vectors for each tetrahedron
    t0 = geometry.t[:, 0]
    t1 = geometry.t[:, 1]
    t2 = geometry.t[:, 2]
    t3 = geometry.t[:, 3]

    v0 = geometry.v[t0, :]
    v1 = geometry.v[t1, :]
    v2 = geometry.v[t2, :]
    v3 = geometry.v[t3, :]

    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0

    # Compute volume of each tetrahedron using triple product formula: V = |(e1 . (e2 x e3))| / 6
    tetra_vols = np.abs(np.einsum('ij,ij->i', e1, np.cross(e2, e3))) / 6

    # Compute centroid of each tetrahedron as simple average of its vertices
    tetra_centroids = (v0 + v1 + v2 + v3) / 4

    # Compute mesh centroid as volume-weighted average of tetrahedron centroids
    # Note: this is equivalent to LaPy's TriaMesh.centroid()
    centroid = np.sum(tetra_vols[:, np.newaxis] * tetra_centroids, axis=0) / np.sum(tetra_vols)

    # Translate centroid to origin
    geometry.v -= centroid

    # Rescale to unit volume
    geometry.v /= geometry.boundary_tria().volume() ** (1/3)

    return geometry

def check_vol(
    vol: TetMesh
) -> None:
    """
    Check if the volume mesh has no unreferenced vertices and a contiguous surface boundary.
    
    Parameters
    ----------
    vol : lapy.TetMesh
        The volume mesh to check.

    Raises
    ------
    ValueError
        If the volume mesh contains unreferenced vertices.
    ValueError
        If the volume mesh is not manifold (i.e., contains triangles shared by more than two
        tetrahedra).
    """
    if vol.has_free_vertices():
        raise ValueError('Volume mesh contains unreferenced vertices (i.e., not part of any '
                         'tetrahedron).')
    
    # Ensure volume is manifold (i.e., no cells shared by more than two tets)
    if not _is_vol_manifold(vol):
        raise ValueError('Volume mesh is not manifold: contains cells shared by more than two '
                         'tetrahedra.')

    # Validate surface boundary
    vol_boundary = vol.boundary_tria()
    vol_boundary.rm_free_vertices_()
    vol_boundary.orient_()
    check_surf(vol_boundary)

def _is_vol_manifold(vol) -> bool:
    """Check if the tetrahedral mesh is manifold.

    Returns
    -------
    bool
        True if every triangle face is shared by at most two tetrahedra.
    """
    # Extract all 4 triangles from each tetrahedron
    trias = np.concatenate([
        vol.t[:, [0, 1, 2]],
        vol.t[:, [0, 1, 3]],
        vol.t[:, [0, 2, 3]],
        vol.t[:, [1, 2, 3]],
    ])

    # Order vertices within each triangle for consistent representation
    trias.sort(axis=1)

    # Manifold if no triangle occurs more than twice
    tria_counts = np.unique(trias, axis=0, return_counts=True)[1]
    return np.all(tria_counts <= 2)

def check_surf(
    surf: TriaMesh
) -> None:
    """
    Check if the surface mesh is contiguous, manifold, and contains no unreferenced vertices.
    
    Parameters
    ----------
    surf : lapy.TriaMesh
        The surface mesh to check.

    Raises
    ------
    ValueError
        If the surface mesh contains unreferenced (i.e., faceless) vertices.
    ValueError
        If the surface mesh is not contiguous.
    ValueError
        If the surface mesh is not manifold (i.e., contains edges belonging to more than two cells).
    """
    # Ensure surface has no unreferenced vertices
    referenced = np.zeros(len(surf.v), dtype=bool)
    referenced[surf.t] = True
    if not np.all(referenced):
        raise ValueError(f'Surface mesh contains {np.sum(~referenced)} unreferenced '
                         '(i.e., faceless) vertices.')

    # Ensure surface is contiguous
    n_components = surf.connected_components()[0]
    if n_components != 1:
        raise ValueError(f'Surface mesh is not contiguous: {n_components} connected components '
                         'found.')

    # Ensure surface is manifold
    if not surf.is_manifold():
        raise ValueError('Surface mesh is not manifold: contains edges belonging to more than two '
                         'cells.')

##################################################################################
##################################################################################
################################ HELPERS #########################################
##################################################################################
##################################################################################

def _load_nifti(nifti: str | Path | Nifti1Image) -> Nifti1Image:
    """
    Load a NIFTI file from a path or return the input if it is already a Nifti1Image.

    Parameters
    ----------
    nifti : str, Path, or Nifti1Image
        Input NIFTI file path or loaded ``Nifti1Image`` object.

    Returns
    -------
    Nifti1Image
        Loaded NIFTI image.

    Raises
    ------
    TypeError
        If ``nifti`` is not a ``Nifti1Image`` object or a path-like string to a valid ``.nii``
        or ``.nii.gz`` file.
    """
    if isinstance(nifti, (str, Path)):
        nifti = Nifti1Image.load(nifti)
    elif not isinstance(nifti, Nifti1Image):
        raise TypeError("nifti must be a Nifti1Image object or a path-like string to a valid "
                        "`.nii` or `.nii.gz` file.")
    return nifti

def _default_volume_mask(volume: NDArray[np.floating]) -> NDArray[np.bool_]:
    return np.any(volume.astype(bool), axis=tuple(range(3, volume.ndim)))

def _mask_to_coords(
    mask: NDArray[np.bool_], 
    affine: NDArray[np.floating] | None = None
) -> NDArray[np.floating]:
    voxel_indices = np.column_stack(np.nonzero(mask))
    return voxel_indices if affine is None else apply_affine(affine, voxel_indices)

def _nifti_to_coords(
    nifti: str | Path | Nifti1Image,
    mask_or_function: NDArray[np.bool_] | _CallableFloatingToBool = _default_volume_mask,
) -> NDArray[np.floating]:
    nifti = _load_nifti(nifti)
    mask = _generate_mask_for_nifti(nifti, mask_or_function)
    return _mask_to_coords(mask, nifti.affine)

def _generate_mask_for_volume(
    volume: NDArray[np.floating],
    mask_or_function: NDArray[np.bool_] | _CallableFloatingToBool = _default_volume_mask,
    # don't use None as it is ambiguous (default mask or no mask?)
) -> NDArray[np.bool_]:
    mask = mask_or_function(volume) if callable(mask_or_function) else mask_or_function
    if mask.ndim != 3 or mask.shape != volume.shape[:3]:
        raise ValueError(f"nifti_mask must be a 3D image with dimensions {volume.shape[:3]}.")
    return mask

def _generate_mask_for_nifti(
    nifti: str | Path | Nifti1Image,
    mask_or_function: NDArray[np.bool_] | _CallableFloatingToBool = _default_volume_mask,
    # don't use None as it is ambiguous (default mask or no mask?)
) -> NDArray[np.bool_]:
    nifti = _load_nifti(nifti)
    return _generate_mask_for_volume(nifti.get_fdata(), mask_or_function)

def _make_vertex_mapper(
    mask: NDArray[np.bool_],
):
    if mask.ndim != 3:
        raise ValueError("mask must be a 3D boolean array.")
    def voxel_to_vertex(volume_data: NDArray[np.floating]) -> NDArray[np.floating]:
        if volume_data.ndim < 3:
            raise ValueError("volume_data must be at least 3D.")
        if volume_data.shape[:3] != mask.shape:
            raise ValueError(f"volume_data must have shape {mask.shape + volume_data.shape[3:]}.")
        return volume_data[mask]
    return voxel_to_vertex

def _voxel_data_to_vertex_space(
    voxel_data: NDArray[np.floating],
    mask: NDArray[np.bool_]
) -> NDArray[np.floating]:
    vertex_mapper = _make_vertex_mapper(mask)
    return vertex_mapper(voxel_data)

def _make_voxel_mapper(
    mask: NDArray[np.bool_]
):
    if mask.ndim != 3:
        raise ValueError("mask must be a 3D boolean array.")
    def vertex_to_voxel(vertex_data: NDArray[np.floating], background: float = np.nan) -> NDArray[np.floating]:
        if vertex_data.ndim < 1:
            raise ValueError("vertex_data must be at least 1D.")
        if vertex_data.shape[0] != np.sum(mask):
            raise ValueError(f"vertex_data must have shape ({np.sum(mask)}, ...) to match the "
                             f"number of True values in mask; got {vertex_data.shape}.")
        out_shape = mask.shape + vertex_data.shape[1:]
        out_dtype = np.result_type(vertex_data.dtype, np.array(background).dtype)
        output = np.full(out_shape, background, dtype=out_dtype)
        output[mask] = vertex_data
        return output
    return vertex_to_voxel

def _vertex_data_to_voxel_space(
    vertex_data: NDArray[np.floating],
    mask: NDArray[np.bool_] | _CallableFloatingToBool = _default_volume_mask,
    background: float = np.nan
) -> NDArray[np.floating]:
    mask = _generate_mask_for_volume(vertex_data, mask)
    voxel_mapper = _make_voxel_mapper(mask)
    return voxel_mapper(vertex_data, background=background)

# TODO P1 read docs on how best to do this ?other data to be copied ?inbuilt function
def _copy_volume_to_nifti(
    volume: NDArray,
    nifti: Nifti1Image,
) -> Nifti1Image:
    header = nifti.header.copy()
    header.set_data_dtype(volume.dtype)
    return Nifti1Image(volume, nifti.affine, header=header)



##################################################################################
##################################################################################
####################### SIMPLE MESHER ############################################
##################################################################################
##################################################################################

_CUBE_CORNERS: NDArray[np.int64] = np.array(
    np.unravel_index(np.arange(8), (2, 2, 2)), dtype=np.int64
).T

# @lru_cache(maxsize=1)
def build_cubic_delaunay_lut() -> tuple[NDArray[np.int32], ...]:
    """
    Precompute conforming 3D Delaunay triangulations for all 256 binary
    corner configurations of a 2x2x2 voxel cell.
    """
    def _det3x3_int(
        a: NDArray[np.int64], b: NDArray[np.int64], c: NDArray[np.int64]
    ) -> NDArray[np.int64]:
        """Compute exact integer 3x3 determinant det([a, b, c]) along the last axis."""
        return (
            a[..., 0] * (b[..., 1] * c[..., 2] - b[..., 2] * c[..., 1])
            - a[..., 1] * (b[..., 0] * c[..., 2] - b[..., 2] * c[..., 0])
            + a[..., 2] * (b[..., 0] * c[..., 1] - b[..., 1] * c[..., 0])
        )

    def _det_active_corners(active: NDArray[np.bool_]) -> int:
        """Compute exact integer 3x3 determinant of the 4 active corners of a cube."""
        curr = _CUBE_CORNERS[active]
        return int(_det3x3_int(curr[1] - curr[0], curr[2] - curr[0], curr[3] - curr[0]))

    def _det_tets(tets: NDArray[np.int32]) -> NDArray[np.int64]:
        """Compute exact integer 3x3 determinant of the 4 corners of each tetrahedron."""
        c = _CUBE_CORNERS[tets]  # (T, 4, 3)
        return _det3x3_int(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0], c[:, 3] - c[:, 0])

    # Off-diagonal linear shear breaks cocircularity/cosphericality while
    # keeping opposite cube faces congruent under translation.
    shear = np.array(
        [
            [1.0, 1.1e-4, 1.3e-4],
            [1.7e-4, 1.0, 1.9e-4],
            [2.3e-4, 2.9e-4, 1.0],
        ],
        dtype=np.float64,
    )
    sheared_corners = _CUBE_CORNERS.astype(np.float64) @ shear

    lut: list[NDArray[np.int32]] = []
    for case_id in range(256):
        active = np.array([(case_id >> (7 - b)) & 1 for b in range(8)], dtype=bool)
        n_active = int(active.sum())
        if n_active < 4 or (n_active == 4 and _det_active_corners(active) == 0):
            lut.append(np.empty((0, 4), dtype=np.int32))
            continue

        # Compute Delaunay with symmetry-broken sheared corners to avoid degenerate simplices
        simplices = Delaunay(sheared_corners[active]).simplices
        local_idx = np.flatnonzero(active)
        tets = local_idx[simplices].astype(np.int32)

        # Filter out zero-volume coplanar simplices using exact coordinates
        det6 = _det_tets(tets)
        valid = det6 != 0
        tets = tets[valid]

        # Canonical positive orientation in exact coordinates
        neg = det6[valid] < 0
        tets[neg, 2:4] = tets[neg, 3:1:-1]
        lut.append(tets)

    return tuple(lut)

def tetrahedralize_mask(
    mask: NDArray[np.bool_],
    affine: NDArray[np.floating] | None = None,
) -> TetMesh:
    """
    Convert a 3D binary voxel mask into a conforming tetrahedral mesh
    using the 256-case cubic Delaunay lookup table.
    """
    if mask.ndim != 3:
        raise ValueError("mask must be a 3D boolean array.")

    n_verts = np.count_nonzero(mask)
    if n_verts < 4:
        raise ValueError("mask must contain at least 4 active voxels.")

    # Dense 0-based vertex ID grid in C-order (matching np.nonzero order)
    vert_idx_grid = np.full(mask.shape, -1, dtype=np.int32)
    vert_idx_grid[mask] = np.arange(n_verts, dtype=np.int32)

    lut = build_cubic_delaunay_lut()
    tets_list: list[NDArray[np.int32]] = []
    for x, y, z in np.ndindex(
        mask.shape[0] - 1, mask.shape[1] - 1, mask.shape[2] - 1
    ):
        cube_mask = mask[x : x + 2, y : y + 2, z : z + 2].ravel()
        case_id = int(np.packbits(cube_mask)[0])
        tets = lut[case_id]
        if tets.size:
            corners = _CUBE_CORNERS[tets]  # (T, 4, 3)
            tets_list.append(
                vert_idx_grid[x + corners[..., 0], y + corners[..., 1], z + corners[..., 2]]
            )

    if not tets_list:
        raise ValueError("No tetrahedra could be generated from the provided mask.")

    tets_array = np.vstack(tets_list)

    # Error if any active voxel did not participate in at least one tetrahedron
    used = np.zeros(n_verts, dtype=bool)
    used[tets_array] = True
    if not np.all(used):
        raise ValueError(
            f"{int((~used).sum())} of {n_verts} active voxels were dropped (not part of any tetrahedron)."
        )

    if affine is not None and np.linalg.det(affine[:3, :3]) < 0:
        tets_array[:, 2:4] = tets_array[:, 3:1:-1]

    return TetMesh(_mask_to_coords(mask, affine), tets_array)

def tetrahedralize_nifti(
    nifti: str | Path | Nifti1Image,
    mask_or_function: NDArray[np.bool_] | _CallableFloatingToBool = _default_volume_mask,
) -> TetMesh:
    """
    Convert a 3D binary voxel mask from a NIFTI file into a conforming tetrahedral mesh
    using the 256-case cubic Delaunay lookup table.
    """
    nifti = _load_nifti(nifti)
    mask = _generate_mask_for_nifti(nifti, mask_or_function)
    return tetrahedralize_mask(mask, nifti.affine)










##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################





def tetmesh_data_to_nifti(
    data: NDArray,
    tetmesh: TetMesh,
    nifti_mask: str | Path | Nifti1Image,
) -> Nifti1Image:
    """
    Place tetrahedral mesh data at corresponding nonzero NIFTI mask voxels. Every nonzero voxel in
    ``nifti_mask`` must have exactly one corresponding vertex in ``tetmesh``. No interpolation is
    performed, and extra mesh vertices are allowed. 

    Parameters
    ----------
    data : numpy.ndarray
        Data to be placed at the corresponding NIFTI mask voxels, of shape ``(n_verts,)`` or
        ``(n_verts, n_maps)``.
    tetmesh : lapy.TetMesh
        The tetrahedral mesh containing the vertices corresponding to the NIFTI mask voxels.
    nifti_mask : str, Path, or Nifti1Image
        Input NIFTI file path or loaded ``Nifti1Image`` object containing a binary mask of the
        region of interest (ROI). Every nonzero voxel in this mask must correspond to exactly one
        vertex in ``tetmesh``.

    Returns
    -------
    Nifti1Image
        A new NIFTI image with the same shape and affine as ``nifti_mask``. For multiple maps,
        the NIFTI has shape ``nifti_mask.shape + (n_maps,)``.

    Raises
    ------
    TypeError
        If ``tetmesh`` is not an instance of ``lapy.TetMesh``.
    ValueError
        If ``data`` does not have shape ``(n_verts,)`` or ``(n_verts, n_maps)``.
    TypeError
        If ``nifti_mask`` is not a ``Nifti1Image`` object or a path-like string to a valid ``.nii``
        or ``.nii.gz`` file.
    ValueError
        If any nonzero voxel in ``nifti_mask`` does not correspond to exactly one vertex in
        ``tetmesh``.
    """
    # Format / validate arguments
    # TODO replace with _check_n_nverts once merged
    if data.ndim not in (1, 2) or data.shape[0] != tetmesh.v.shape[0]:
        raise ValueError("data must have shape (n_verts,) or (n_verts, n_maps), where "
                         f"n_verts is {tetmesh.v.shape[0]}; got {data.shape}.")

    nifti_mask = _load_nifti(nifti_mask)

    # Main
    # TODO P1 have a better way of getting the mask (rather than just assuming b/g is 0)
    # What if it should be a given ROI, or a given threshold, or a given set of labels?
    # A `mask_function` that is a Callable that takes the nifti_fdata and returns a boolean mask?
    volume = _map_data_to_volume(data, tetmesh.v, nifti_mask.get_fdata().astype(bool), nifti_mask.affine)
    return _copy_volume_to_nifti(volume, nifti_mask)

def _map_data_to_volume(
    data: NDArray, # TODO type hint that output is same type as input
    vertices: NDArray[np.floating],
    volume: NDArray[np.bool_],
    affine: NDArray[np.floating] | None,
) -> NDArray:
    assert data.shape[0] == vertices.shape[0]
    voxel_indices = np.column_stack(np.nonzero(volume))
    voxel_coords = apply_affine(affine, voxel_indices)
    _, nearest_vertices = KDTree(vertices).query(voxel_coords)

    # Construct NIFTI via direct sampling of mesh data
    # TODO P2 look for better way of setting background
    output = np.zeros(volume.shape + data.shape[1:], dtype=data.dtype)
    output[tuple(voxel_indices.T)] = data[nearest_vertices]
    return output




# TODO consider if the following three functions are needed/useful and delete/uncomment as desired
# def _map_vertices_to_volume(
#     vertices: NDArray[np.floating],
#     volume: NDArray[np.bool_],
#     affine: NDArray[np.floating] | None,
# ) -> NDArray[np.intp]:
#     return _map_data_to_volume(np.arange(1, vertices.shape[0] + 1, dtype=np.intp), 
#                                vertices, volume, affine)

# def _map_data_to_nifti(
#     data: NDArray[np.floating],
#     vertices: NDArray[np.floating],
#     nifti_mask: Nifti1Image,
# ) -> Nifti1Image:
#     nifti_data = _map_data_to_volume(data, vertices, nifti_mask.get_fdata().astype(bool), nifti_mask.affine)
#     return _copy_data_to_nifti(nifti_data, nifti_mask)

# def _map_vertices_to_nifti(
#     vertices: NDArray[np.floating],
#     nifti_mask: Nifti1Image,
# ) -> Nifti1Image:
#     nifti_data = _map_vertices_to_volume(vertices, nifti_mask.get_fdata().astype(bool), nifti_mask.affine)
#     return _copy_data_to_nifti(nifti_data, nifti_mask)



# TODO: consider replacing linear interpolation and RBF with biharmonic inpainting
def nifti_data_to_tetmesh(
    nifti_data: str | Path | Nifti1Image,
    tetmesh: TetMesh,
    nifti_mask: str | Path | Nifti1Image | NDArray[np.bool_] | _CallableFloatingToBool = _default_volume_mask,
    **rbf_kwargs
) -> NDArray[np.floating]:
    """
    Project one or more maps from volumetric NIFTI space to a tetrahedral mesh using linear
    interpolation and radial basis functions for extrapolation.
    
    Parameters
    ----------
    nifti_data : str, Path, or Nifti1Image
        Input NIFTI file path or loaded ``Nifti1Image`` object.
    tetmesh : lapy.TetMesh
        The tetrahedral mesh to which data is projected.
    nifti_mask : str, Path, or Nifti1Image or NDArray[np.bool_] or _CallableFloatingToBool
        Optional mask to define the region of interest. If ``None``, nonzero voxels are used.
        Default is ``None``.
    **rbf_kwargs
        Additional keyword arguments to pass to ``scipy.interpolate.RBFInterpolator``, such as 
        ``function`` and ``smooth``.
    
    Returns
    -------
    ndarray
    Data interpolated at mesh vertices, shape ``(n_vertices,)`` for one map or
    ``(n_vertices, n_maps)`` for a 4D NIFTI.

    Raises
    ------
    TypeError
        If ``nifti_data`` is not a ``Nifti1Image`` object or a path-like string to a valid ``.nii``
        or ``.nii.gz`` file.
    TypeError
        If ``tetmesh`` is not an instance of ``lapy.TetMesh``.
    TypeError
        If ``nifti_mask`` is not a ``Nifti1Image`` object or a path-like string to a valid ``.nii``
        or ``.nii.gz`` file.
    ValueError
        If the dimensions of ``nifti_mask`` do not match those of ``nifti_data``.
    ValueError
        If ``nifti_data`` does not contain a 3D volume or a 4D series of maps.
    """
    if not isinstance(tetmesh, TetMesh):
        raise TypeError("tetmesh must be an instance of lapy.TetMesh.")

    # Load NIFTI if needed
    nifti_data = _load_nifti(nifti_data)
    
    # Get mask
    data = nifti_data.get_fdata()
    if (is_single_map := data.ndim == 3):
        data = data[:, :, :, None]  # Add a singleton dimension for consistent handling of 4D data
    elif data.ndim != 4:
        raise ValueError("nifti_data must contain a 3D volume or a 4D series of maps.")

    if isinstance(nifti_mask, (np.ndarray, Callable)): 
        mask = _generate_mask_for_volume(data, nifti_mask)
    else: 
        mask = _generate_mask_for_volume(data, _generate_mask_for_nifti(nifti_mask))
    
    # Get voxel coordinates and values
    vox_coords = _mask_to_coords(mask, nifti_data.affine)
    values = data[mask]
    
    # Linear interpolation within convex hull of voxel coordinates
    # TODO P1 the convex hull is too broad. why not the surface of the mesh/mask?
    # TODO: consider adding a mapping parameter to use direct assignment for voxel-matching vertices
    
    # interp_data = griddata(vox_coords, values, tetmesh.v, method='linear')
    # if interp_data.ndim == 1:
    #     interp_data = interp_data[:, None]

    lin = LinearNDInterpolator(vox_coords, values)
    interp_data = lin(tetmesh.v)

    # Use radial basis functions to interpolate where vertices are outside the convex hull
    nan_rows = np.any(np.isnan(interp_data), axis=1)
    if np.any(nan_rows):
        rbf = RBFInterpolator(vox_coords, values, **rbf_kwargs)
        rbf_data = rbf(tetmesh.v[nan_rows])
        interp_data[nan_rows] = np.where(
            np.isnan(interp_data[nan_rows]),
            rbf_data,
            interp_data[nan_rows],
        )

    return interp_data[:, 0] if is_single_map else interp_data

def nifti_mask_to_tetmesh(
    nifti_mask: str | Path | Nifti1Image,
    closings: int = 0,
    discard_components: bool = False,
    **tetgen_kwargs
) -> TetMesh:
    """
    Tetrahedral meshing from a binary voxel mask NIFTI via marching cubes and Delaunay
    tetrahedralization. Note that mesh quality may be poor when the NIFTI has nighly anisotropic
    voxel dimensions. 

    Parameters
    ----------
    nifti_mask : str, Path, or Nifti1Image
        Input NIFTI file path or loaded Nifti1Image object containing a binary mask of the region of
        interest (ROI).
    closings : int, optional
        Number of binary closing operations to perform on the ROI mask before meshing. This can help
        fill small holes and merge disconnected pieces in the ROI. Default is ``0``.
    discard_components : bool, optional
        If True, only the largest connected component of the generated surface mesh will be kept, 
        discarding the rest. This can be useful for removing small, irrelevant pieces from the mesh.
        Default is ``False``.
    **tetgen_kwargs
        Additional keyword arguments to pass to the `tetgen.TetGen.tetrahedralize`.

    Returns
    -------
    lapy.TetMesh
        The generated tetrahedral mesh of the ROI.

    Raises
    ------
    ImportError
        If the required packages `skimage` and `tetgen` are not installed.
    TypeError
        If ``nifti_mask`` is not a Nifti1Image object or a path-like string to a valid ``.nii`` or
        ``.nii.gz``.
        file.
    ValueError
        If ``closings`` is not a non-negative integer.
    ValueError
        If the generated surface mesh from marching cubes is not closed.
    ValueError
        If the generated surface mesh from marching cubes is not manifold. 
    """
    # Handle optional dependencies EAFP style
    try:
        from skimage.measure import marching_cubes
        from tetgen import TetGen
    except ImportError as e:
        raise ImportError("The `skimage` and `tetgen` packages are required for this function. "
                          "Please ensure they are installed in your environment (e.g., pip install "
                          "neuromodes[vol]).") from e

    # Format / validate arguments
    if closings != int(closings) or closings < 0:
        raise ValueError("Parameter `closings` must be a non-negative integer.")
    
    # Get binary ROI from NIFTI
    nifti_mask = _load_nifti(nifti_mask)
    roi = nifti_mask.get_fdata().astype(bool)
    if closings:
        roi = binary_closing(roi, iterations=closings)

    # Marching cubes to extract smoothed surface (replacing mri_mc from FreeSurfer)
    surf_verts, trias, _, _ = marching_cubes(roi, level=0.5, allow_degenerate=False)
    surf_verts = apply_affine(nifti_mask.affine, surf_verts)
    surf = TriaMesh(v=surf_verts, t=trias)
    
    # Handle disconnected components in generated surface
    n_components, labels = surf.connected_components()
    if n_components > 1:
        if discard_components:
            surf.keep_largest_connected_component_()
            print(f"Components discarded: {n_components-1}.")
            unique, counts = np.unique(labels, return_counts=True)
            for comp, count in zip(unique, counts):
                if comp != np.argmax(np.bincount(labels)):
                    print(f"    - {count} vertices.")
        else:
            warn(f"Generated surface has {n_components} connected components; unable to proceed "
                 "with tetrahedral meshing. Surface mesh will be returned with labels array to "
                 "allow visual inspection. Consider using `discard_components` to keep only the "
                 "largest component, or `closings` to fill small holes and merge disconnected "
                 "pieces.")
            return surf, labels

    # Ensure that surface is closed and manifold
    if not surf.is_closed():
        raise ValueError("Generated surface mesh is not closed. Consider using `closings` to fill "
                         "small holes and merge disconnected pieces.")
    if not surf.is_manifold():
        raise ValueError("Generated surface mesh is not manifold: contains edges belonging to more "
                         "than two cells. Consider using `closings` to fill small holes and merge "
                         "disconnected pieces.")
        
    # Append vertices from marching cubes with voxel centers
    vox_coords = np.column_stack(np.nonzero(roi))
    vox_coords = apply_affine(nifti_mask.affine, vox_coords)

    # Generate edges to complete tetrahedral mesh
    tetgen = TetGen(np.vstack([surf.v, vox_coords]), surf.t)
    tetgen.tetrahedralize(**tetgen_kwargs)

    # Convert to lapy
    mesh = TetMesh(v=tetgen.node, t=tetgen.elem)

    # Ensure the generated volume mesh is valid
    check_vol(mesh)

    return mesh
