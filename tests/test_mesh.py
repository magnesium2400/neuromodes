from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
import pytest
from lapy import TetMesh, TriaMesh
from nibabel import Nifti1Image
from nibabel.affines import apply_affine

from neuromodes.mesh import (
    check_surf,
    check_vol,
    mask_mesh,
    nifti_data_to_tetmesh,
    nifti_mask_to_tetmesh,
    tetmesh_data_to_nifti,
    _make_vertex_mapper,
    _make_voxel_mapper,
    build_cubic_delaunay_lut,
    tetrahedralize_mask,
    tetrahedralize_nifti,
    _CUBE_CORNERS
)

DATA_DIR = Path(__file__).parent.parent / "neuromodes" / "data"

_SHAPE_FIXTURES = (
    "mask_cube",
    "mask_cuboid",
    "mask_l_cuboid",
    "mask_octahedron",
    "mask_menger_sponge",
    "mask_sphere",
    "mask_ellipsoid",
    "mask_trefoil_knot",
    "mask_dumbbell",
)



def _tetmesh_volume(mesh: TetMesh) -> float:
    """Compute total signed volume of a tetrahedral mesh."""
    c = np.asarray(mesh.v, dtype=np.float64)[np.asarray(mesh.t)]
    det6 = np.linalg.det(
        np.stack([c[:, 1] - c[:, 0], c[:, 2] - c[:, 0], c[:, 3] - c[:, 0]], axis=-1)
    )
    return float(det6.sum() / 6.0)

class TestVtoV: 
    @pytest.mark.parametrize("shape", [(), (1,), (2,), (1, 1), (1, 2), (2, 1), (2, 2)])
    def test_vertex_to_voxel_to_vertex(self, shape):
        mask = np.random.rand(10, 20, 30) > 0.5
        vertex_data = np.random.rand(mask.sum(), *shape)
        voxelise = _make_voxel_mapper(mask)
        vertexise = _make_vertex_mapper(mask)
        assert np.array_equal(
            vertexise(voxelise(vertex_data)),
            vertex_data,
        )

    @pytest.mark.parametrize("shape", [(), (1,), (2,), (1, 1), (1, 2), (2, 1), (2, 2)])
    def test_voxel_to_vertex_to_voxel(self, shape): 
        mask = np.random.rand(10, 20, 30) > 0.5
        voxel_data = (mask.T * np.random.rand(*mask.shape, *shape).T).T
        voxelise = _make_voxel_mapper(mask)
        vertexise = _make_vertex_mapper(mask)
        assert np.array_equal(
            voxelise(vertexise(voxel_data), background=0.0),
            voxel_data,
        )

class TestCubicLUT:
    @pytest.fixture(scope="class")
    @classmethod
    def lut(cls):
        return build_cubic_delaunay_lut()

    def test_lut_shape(self, lut):
        assert isinstance(lut, tuple)
        assert len(lut) == 256

    def test_tets_shape(self, lut):
        for i, tets in enumerate(lut):
            assert isinstance(tets, np.ndarray)
            assert tets.ndim == 2
            assert tets.shape[1] == 4

            i_binary = np.array(list(np.binary_repr(i, width=8)), dtype=int)
            active_verts = np.flatnonzero(i_binary)
            assert np.all(np.isin(tets, active_verts))

            n_verts = active_verts.size
            n_tets = tets.shape[0]
            match n_verts:
                case 0 | 1 | 2 | 3:
                    assert n_tets == 0
                case 4:
                    assert n_tets in (0, 1)
                case 5:
                    assert n_tets in (2, 3)
                case 6:
                    assert n_tets in (3, 4)
                case 7:
                    assert n_tets == 5
                case 8:
                    assert n_tets == 6

    def test_tets_volume_and_orientation(self, lut):
        for i, tets in enumerate(lut):
            if not tets.size: 
                continue

            i_binary = np.array(list(np.binary_repr(i, width=8)), dtype=int)
            active_verts = np.flatnonzero(i_binary)

            c = _CUBE_CORNERS[tets] # (T, 4, 3)
            det6 = np.linalg.det(
                np.stack([c[:, 1] - c[:, 0], c[:, 2] - c[:, 0], c[:, 3] - c[:, 0]], axis=-1)
            )
            assert np.isclose(det6.sum() / 6.0, ConvexHull(_CUBE_CORNERS[active_verts]).volume)
            assert np.all(np.rint(det6) >= 1)

    def test_diagonal_orientation(self, lut):
        # Store canonical 2D face triangulations on each of the 6 cube faces across all cases
        face_diags: dict[tuple[int, int], set[tuple[tuple[int, int], tuple[int, int]]]] = {
            (axis, val): set() for axis in range(3) for val in (0, 1)
        }

        for i, tets in enumerate(lut):
            if not tets.size: 
                continue

            i_binary = np.array(list(np.binary_repr(i, width=8)), dtype=int)

            # Record face diagonals only when all 4 corners of that face are active
            c = _CUBE_CORNERS[tets]  # (T, 4, 3)
            for tet_coords in c:
                for u in range(4):
                    for v in range(u):
                        p, q = tet_coords[u], tet_coords[v]
                        if np.sum(p != q) == 2:
                            shared_axis = int(np.flatnonzero(p == q)[0])
                            shared_val = int(p[shared_axis])
                            face_active = i_binary[_CUBE_CORNERS[:, shared_axis] == shared_val]
                            if not np.all(face_active):
                                continue

                            other_axes = [a for a in range(3) if a != shared_axis]
                            pt1 = (int(p[other_axes[0]]), int(p[other_axes[1]]))
                            pt2 = (int(q[other_axes[0]]), int(q[other_axes[1]]))
                            edge_2d = (pt1, pt2) if pt1 <= pt2 else (pt2, pt1)
                            face_diags[(shared_axis, shared_val)].add(edge_2d)

        # For each axis, whenever all 4 vertices are selected,
        for axis in range(3):
            # only one diagonal is used
            assert len(face_diags[(axis, 0)]) == 1
            # and it is the same diagonal for val=0 and val=1
            assert face_diags[(axis, 0)] == face_diags[(axis, 1)]

class TestTetrahedralizeMask:
    @pytest.fixture(scope="class")
    @classmethod
    def mask_cube(cls) -> NDArray[np.bool_]:
        mask = np.ones((5, 5, 5), dtype=np.bool_)
        return mask

    @pytest.fixture(scope="class")
    @classmethod
    def mask_cuboid(cls) -> NDArray[np.bool_]:
        mask = np.zeros((7, 11, 8), dtype=np.bool_)
        mask[1:5, 1:10, 1:7] = True
        return mask

    @pytest.fixture(scope="class")
    @classmethod
    def mask_l_cuboid(cls) -> NDArray[np.bool_]:
        mask = np.zeros((11, 11, 7), dtype=np.bool_)
        mask[1:10, 1:4, 1:6] = True
        mask[1:4, 1:10, 1:6] = True
        return mask

    @pytest.fixture(scope="class")
    @classmethod
    def mask_octahedron(cls) -> NDArray[np.bool_]:
        ox, oy, oz = np.meshgrid(
            np.arange(-4, 5), np.arange(-4, 5), np.arange(-4, 5), indexing="ij", sparse=True
        )
        return np.asarray((np.abs(ox) + np.abs(oy) + np.abs(oz)) <= 3, dtype=np.bool_)

    @pytest.fixture(scope="class")
    @classmethod
    def mask_menger_sponge(cls) -> NDArray[np.bool_]:
        depth, length = 2, 2
        n = (3**depth) * length
        sponge = np.ones((n, n, n), dtype=np.bool_)
        all_idx = np.ones(n, dtype=np.bool_)
        for d in range(depth):
            step = (3**d) * length
            mid = (np.arange(n) // step) % 3 == 1
            sponge[np.ix_(mid, mid, all_idx)] = False
            sponge[np.ix_(mid, all_idx, mid)] = False
            sponge[np.ix_(all_idx, mid, mid)] = False
        return sponge

    @pytest.fixture(scope="class")
    @classmethod
    def mask_sphere(cls) -> NDArray[np.bool_]:
        xg, yg, zg = np.meshgrid(
            np.arange(-8, 9), np.arange(-8, 9), np.arange(-8, 9), indexing="ij", sparse=True
        )
        mask: NDArray[np.bool_] = (xg**2 + yg**2 + zg**2) <= 6.5**2
        return mask

    @pytest.fixture(scope="class")
    @classmethod
    def mask_ellipsoid(cls) -> NDArray[np.bool_]:
        xe, ye, ze = np.meshgrid(
            np.arange(-11, 12), np.arange(-8, 9), np.arange(-6, 7), indexing="ij", sparse=True
        )
        mask: NDArray[np.bool_] = ((xe / 9.2) ** 2 + (ye / 6.2) ** 2 + (ze / 4.2) ** 2) <= 1.0
        return mask

    @pytest.fixture(scope="class")
    @classmethod
    def mask_trefoil_knot(cls) -> NDArray[np.bool_]:
        t = np.linspace(0.0, 2.0 * np.pi, 600, endpoint=False)
        centerline = np.column_stack(
            [np.sin(t) + 2.0 * np.sin(2.0 * t), np.cos(t) - 2.0 * np.cos(2.0 * t), -np.sin(3.0 * t)]
        )
        kx, ky, kz = np.meshgrid(
            np.linspace(-4.2, 4.2, 26),
            np.linspace(-4.2, 4.2, 26),
            np.linspace(-2.2, 2.2, 16),
            indexing="ij",
        )
        pts = np.column_stack([kx.ravel(), ky.ravel(), kz.ravel()])
        dist = np.min(np.linalg.norm(pts[:, None, :] - centerline[None, :, :], axis=-1), axis=1)
        mask: NDArray[np.bool_] = (dist <= 0.5).reshape(kx.shape)
        return mask

    @pytest.fixture(scope="class")
    @classmethod
    def mask_dumbbell(cls) -> NDArray[np.bool_]:
        bell_radius, neck_radius, neck_half_length, pad = 5.5, 2.0, 6.0, 1
        bell_offset = neck_half_length + 0.65 * bell_radius
        rx = int(np.ceil(bell_offset + bell_radius)) + pad
        ryz = int(np.ceil(bell_radius)) + pad
        x, y, z = np.meshgrid(
            np.arange(-rx, rx + 1, dtype=np.float64),
            np.arange(-ryz, ryz + 1, dtype=np.float64),
            np.arange(-ryz, ryz + 1, dtype=np.float64),
            indexing="ij",
            sparse=True,
        )
        r_yz2 = y**2 + z**2
        mask: NDArray[np.bool_] = (
            (((x + bell_offset) ** 2 + r_yz2) <= bell_radius**2)
            | (((x - bell_offset) ** 2 + r_yz2) <= bell_radius**2)
            | ((np.abs(x) <= bell_offset) & (r_yz2 <= neck_radius**2))
        )
        return mask

    @pytest.fixture(scope="class", params=_SHAPE_FIXTURES)
    @classmethod
    def mask_and_mesh(cls, request: pytest.FixtureRequest) -> tuple[NDArray[np.bool_], TetMesh]:
        """Meshes each shape once per class for reuse across all invariant tests."""
        mask: NDArray[np.bool_] = request.getfixturevalue(request.param)
        mesh = tetrahedralize_mask(mask)
        check_vol(mesh)
        return mask, mesh

    def test_n_verts(self, mask_and_mesh: tuple[NDArray[np.bool_], TetMesh]) -> None:
        mask, mesh = mask_and_mesh
        assert np.count_nonzero(mask) == mesh.v.shape[0]

    def test_positive_jacobians(self, mask_and_mesh: tuple[NDArray[np.bool_], TetMesh]) -> None:
        _, mesh = mask_and_mesh
        c = np.asarray(mesh.v)[np.asarray(mesh.t)]
        det6 = np.linalg.det(
            np.stack([c[:, 1] - c[:, 0], c[:, 2] - c[:, 0], c[:, 3] - c[:, 0]], axis=-1)
        )
        assert np.all(det6 > 0.0)

    def test_conforming_faces(self, mask_and_mesh: tuple[NDArray[np.bool_], TetMesh]) -> None:
        _, mesh = mask_and_mesh
        t = np.asarray(mesh.t)
        raw_faces = t[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3)
        _, counts = np.unique(np.sort(raw_faces, axis=1), axis=0, return_counts=True)
        assert np.all((counts == 1) | (counts == 2))
    
    @pytest.mark.parametrize(
        "mask_name",
        ["mask_cube", "mask_cuboid", "mask_octahedron"],
    )
    @pytest.mark.parametrize("use_las_affine", [False, True])
    def test_convex_volume_matches_convex_hull(
        self,
        request: pytest.FixtureRequest,
        mask_name: str,
        use_las_affine: bool,
    ) -> None:
        mask: NDArray[np.bool_] = request.getfixturevalue(mask_name)
        affine = None
        if use_las_affine:
            affine = np.diag([-0.8, 1.2, 1.8, 1.0])
            affine[:3, 3] = [12.0, -6.0, 3.0]

        mesh = tetrahedralize_mask(mask, affine=affine)
        tet_vol = _tetmesh_volume(mesh)
        hull_vol = float(ConvexHull(mesh.v).volume)

        assert np.isclose(tet_vol, hull_vol, rtol=1e-12, atol=1e-12)

    def test_volume_bounded_by_convex_hull(
        self, mask_and_mesh: tuple[NDArray[np.bool_], TetMesh]
    ) -> None:
        _, mesh = mask_and_mesh
        tet_vol = _tetmesh_volume(mesh)
        hull_vol = float(ConvexHull(mesh.v).volume)
        assert 0.0 < tet_vol <= hull_vol + 1e-12

    @pytest.mark.parametrize("mask_name", _SHAPE_FIXTURES)
    def test_las_affine_orientation(
        self, request: pytest.FixtureRequest, mask_name: str
    ) -> None:
        mask: NDArray[np.bool_] = request.getfixturevalue(mask_name)
        affine_las = np.diag([-0.8, 1.2, 1.8, 1.0])
        affine_las[:3, 3] = [12.0, -6.0, 3.0]

        mesh = tetrahedralize_mask(mask, affine=affine_las)
        c = np.asarray(mesh.v)[np.asarray(mesh.t)]
        det6 = np.linalg.det(
            np.stack([c[:, 1] - c[:, 0], c[:, 2] - c[:, 0], c[:, 3] - c[:, 0]], axis=-1)
        )
        assert np.all(det6 > 0.0)

class TestTetrahedralizeMaskHipTha: 
    @pytest.mark.parametrize('subcort', ['hippo', 'tha'])
    def test_tetrahedralize_mask_empirical(self, subcort: str):
        """
        Test that tetrahedralize_mask produces the same mesh as tetrahedralize_nifti
        for the included hippocampus and thalamus masks, without errors
        """
        mask_path = DATA_DIR / f'hcp_{subcort}-lh_thr25.nii.gz'

        mesh1 = tetrahedralize_nifti(mask_path)

        mask_image = Nifti1Image.load(mask_path)
        mask = mask_image.get_fdata() != 0
        affine = mask_image.affine

        mesh2 = tetrahedralize_mask(mask, affine=affine)

        np.testing.assert_allclose(mesh1.v, mesh2.v)
        np.testing.assert_array_equal(mesh1.t, mesh2.t)

        # Check that the mesh is valid
        check_vol(mesh2)

@pytest.fixture(scope="module")
def mask_meshes():
    return {
        name: nifti_mask_to_tetmesh(DATA_DIR / name)
        for name in ('hcp_tha-lh_thr25.nii.gz', 'hcp_hippo-lh_thr25.nii.gz')
    }

def _vox_coords(image):
    voxel_indices = np.column_stack(np.nonzero(image.get_fdata()))
    return apply_affine(image.affine, voxel_indices)

def test_nifti_mask_to_tetmesh(mask_meshes):
    for name, mesh in mask_meshes.items():
        image = Nifti1Image.load(DATA_DIR / name)
        voxel_coords = _vox_coords(image)

        # check that all nonzero voxels are represented in the mesh
        assert all(
            np.any(np.all(np.isclose(mesh.v, voxel_coord), axis=1))
            for voxel_coord in voxel_coords
        )

        # check that all mesh vertices are within one voxel's width of a nonzero voxel
        vox_width = image.header.get_zooms()[0]  # isotropic; 2 (mm) for included meshes
        voxel_coords_index = apply_affine(np.linalg.inv(image.affine), voxel_coords)
        mesh_coords_index = apply_affine(np.linalg.inv(image.affine), mesh.v)
        nearest_distances = np.min(
            np.linalg.norm(
                mesh_coords_index[:, None, :] - voxel_coords_index[None, :, :],
                axis=2,
            ),
            axis=1,
        )
        assert np.max(nearest_distances) <= vox_width

def test_nifti_mask_to_tetmesh_faulty_return():

    with pytest.warns(UserWarning, match="2 connected components"):
        mesh, labels = nifti_mask_to_tetmesh(DATA_DIR / "hcp_striatum-lh_thr25.nii.gz")

    assert isinstance(mesh, TriaMesh)
    assert labels.shape == (mesh.v.shape[0],)
    assert np.unique(labels).size == 2

def test_nifti_mask_to_tetmesh_faulty_repair():
    path = DATA_DIR / "hcp_striatum-lh_thr25.nii.gz"

    # just check that either binary closing or discarding components runs without error
    nifti_mask_to_tetmesh(path, closings=1)
    nifti_mask_to_tetmesh(path, discard_components=True)

def test_nifti_data_to_tetmesh(mask_meshes):
    # Load hippocampus mask
    image = Nifti1Image.load(DATA_DIR / "hcp_hippo-lh_thr25.nii.gz")
    mask = image.get_fdata() != 0
    voxel_indices = np.column_stack(np.nonzero(mask))

    # Set data to physical-space x-coordinates
    data = np.zeros(image.shape)
    voxel_coords = apply_affine(image.affine, voxel_indices)
    data[mask] = voxel_coords[:, 0]
    data_image = Nifti1Image(data, image.affine, image.header)

    # Interpolate to tetmesh
    mesh = mask_meshes["hcp_hippo-lh_thr25.nii.gz"]
    mesh_data = nifti_data_to_tetmesh(data_image, mesh, image)

    # check that all nonzero voxels are represented in the mesh
    for voxel_coord in _vox_coords(image):
        vertex = np.flatnonzero(np.all(np.isclose(mesh.v, voxel_coord), axis=1))
        assert vertex.size == 1
        assert np.isclose(mesh_data[vertex[0]], voxel_coord[0])

    # check that interpolated data matches physical mesh vertex coordinates
    assert np.allclose(mesh_data, mesh.v[:, 0], atol=1e-6)

def test_nifti_data_to_tetmesh_multiple_maps(mask_meshes):
    # Load hippocampus mask and mesh
    hippo = "hcp_hippo-lh_thr25.nii.gz"
    image = Nifti1Image.load(DATA_DIR / hippo)
    mesh = mask_meshes[hippo]
    mask = image.get_fdata() != 0

    # Create data with two maps: physical-space x- and y-coordinates
    voxel_indices = np.column_stack(np.nonzero(mask))
    voxel_coords = apply_affine(image.affine, voxel_indices)
    data = np.zeros(image.shape + (2,))
    data[mask, 0] = voxel_coords[:, 0]
    data[mask, 1] = voxel_coords[:, 1]
    data_image = Nifti1Image(data, image.affine, image.header)

    # Interpolate to tetmesh and check that it matches NIFTI data
    mesh_data = nifti_data_to_tetmesh(data_image, mesh, image)
    assert mesh_data.shape == (mesh.v.shape[0], 2)
    assert np.allclose(mesh_data[:, 0], mesh.v[:, 0], atol=1e-6)
    assert np.allclose(mesh_data[:, 1], mesh.v[:, 1], atol=1e-6)

def test_tetmesh_data_to_nifti(mask_meshes):
    # Load hippocampus mask and mesh
    hippo = "hcp_hippo-lh_thr25.nii.gz"
    image = Nifti1Image.load(DATA_DIR / hippo)
    mesh = mask_meshes[hippo]

    # Create data on mesh and convert back to NIFTI
    data = np.arange(mesh.v.shape[0], dtype=float)
    output = tetmesh_data_to_nifti(data, mesh, image)

    # check that resampled data is a direct sampling of the mesh data
    voxel_indices = np.column_stack(np.nonzero(image.get_fdata()))
    voxel_coords = apply_affine(image.affine, voxel_indices)
    distances = np.linalg.norm(
        mesh.v[:, None, :] - voxel_coords[None, :, :],
        axis=2,
    )
    mesh_vertices = np.argmin(distances, axis=0)

    assert output.shape == image.shape
    assert np.array_equal(output.get_fdata()[tuple(voxel_indices.T)], data[mesh_vertices])

def test_tetmesh_data_to_nifti_multiple_maps(mask_meshes):
    # Load hippocampus mask and mesh
    hippo = "hcp_hippo-lh_thr25.nii.gz"
    image = Nifti1Image.load(DATA_DIR / hippo)
    mesh = mask_meshes[hippo]

    # Create data on mesh and convert back to NIFTI
    data = np.column_stack((mesh.v[:, 0], mesh.v[:, 1]))
    output = tetmesh_data_to_nifti(data, mesh, image)

    # check that resampled data is a direct sampling of the mesh data
    voxel_indices = np.column_stack(np.nonzero(image.get_fdata()))
    voxel_coords = apply_affine(image.affine, voxel_indices)
    distances = np.linalg.norm(
        mesh.v[:, None, :] - voxel_coords[None, :, :],
        axis=2,
    )
    mesh_vertices = np.argmin(distances, axis=0)

    assert output.shape == image.shape + (2,)
    assert np.allclose(
        output.get_fdata()[tuple(voxel_indices.T)],
        data[mesh_vertices],
    )

def test_tetmesh_data_to_nifti_aniso_affine():
    # Load hippocampus mask
    hippo = "hcp_hippo-lh_thr25.nii.gz"
    image = Nifti1Image.load(DATA_DIR / hippo)

    # Make affine anisotropic and create new mesh from it
    aniso_affine = image.affine.copy()
    aniso_affine[:3, :3] = aniso_affine[:3, :3] @ np.diag([1.5, 2.0, 0.75])
    aniso_mask = Nifti1Image(image.get_fdata(), aniso_affine, image.header)
    mesh = nifti_mask_to_tetmesh(aniso_mask)

    # Create data on mesh and convert back to NIFTI
    data = np.arange(mesh.v.shape[0], dtype=float)
    output = tetmesh_data_to_nifti(data, mesh, aniso_mask)

    # check that resampled data is a direct sampling of the mesh data
    voxel_indices = np.column_stack(np.nonzero(image.get_fdata()))
    voxel_coords = apply_affine(aniso_affine, voxel_indices)
    distances = np.linalg.norm(
        mesh.v[:, None, :] - voxel_coords[None, :, :],
        axis=2,
    )
    mesh_vertices = np.argmin(distances, axis=0)

    assert np.array_equal(
        output.get_fdata()[tuple(voxel_indices.T)],
        data[mesh_vertices],
    )

# def test_tetmesh_data_to_nifti_unmatched_voxels():
#     # Create a mask with voxels that do not have corresponding vertices in the mesh
#     mask = np.zeros((2, 2, 2))
#     mask[0, 0, 0] = 1
#     mask[1, 0, 0] = 1
#     image = Nifti1Image(mask, np.eye(4))
#     mesh = TetMesh(
#         v=np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float),
#         t=np.array([[0, 1, 2, 3]]),
#     )

#     with pytest.raises(ValueError, match="must correspond to a unique tetmesh vertex"):
#         tetmesh_data_to_nifti(np.arange(4), mesh, image)

def test_mask_surf():

    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1]
    ])

    cells = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [1, 2, 4],
        [1, 3, 5]
    ])

    vol = TriaMesh(v=verts, t=cells)

    mask = np.array([True, True, True, True, False, False])

    masked_vol = mask_mesh(vol, mask)

    # Only the first tetrahedron should remain
    assert masked_vol.v.shape[0] == 4
    assert (masked_vol.t == [[0, 1, 2], [0, 2, 3]]).all()

def test_surf_unreferenced_verts():
    # Create an invalid surface mesh with unreferenced vertices
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 2]])  # Last vertex unreferenced
    cells = np.array([[0, 1, 2], [0, 2, 3]])  # Only uses first 4 vertices, vertex 4 is unreferenced
    
    invalid_mesh = TriaMesh(v=vertices, t=cells)
   
    # check_surf should raise ValueError due to unreferenced vertex
    with pytest.raises(ValueError, match="Surface mesh contains .* unreferenced"):
        check_surf(invalid_mesh)

def test_surf_not_contiguous():
    # Create two separate triangles (disconnected components)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]])
    cells = np.array([[0, 1, 2], [3, 4, 5]])  # Two separate triangles
    disconnected_mesh = TriaMesh(v=vertices, t=cells)
    
    # check_surf should raise ValueError due to multiple components
    with pytest.raises(ValueError, match="Surface mesh is not contiguous.*connected components"):
        check_surf(disconnected_mesh)

def test_mask_vol():
    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
    ])

    tets = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [1, 3, 4, 5],
    ])

    vol = TetMesh(v=verts, t=tets)

    mask = np.array([True, True, True, True, False, False])

    masked_vol = mask_mesh(vol, mask)

    # Only the first tetrahedron should remain
    assert masked_vol.v.shape[0] == 4
    assert (masked_vol.t == [0, 1, 2, 3]).all()

def test_vol_unreferenced_verts():
    # Create an invalid volume mesh with unreferenced vertices
    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [2, 2, 2]  # Unreferenced vertex
    ])

    tets = np.array([
        [0, 1, 2, 3]
    ])

    vol = TetMesh(v=verts, t=tets)
   
    # check_vol should raise ValueError due to unreferenced vertex
    with pytest.raises(ValueError, match="Volume mesh contains unreferenced vertices"):
        check_vol(vol)

def test_vol_boundary_not_contiguous():
    # Create a volume mesh of two disconnected tetrahedra
    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [2, 0, 0],
        [3, 0, 0],
        [2, 1, 0],
        [2, 0, 1]
    ])

    tets = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7]
    ])

    vol = TetMesh(v=verts, t=tets)
    
    # check_vol should raise ValueError due to multiple components
    with pytest.raises(ValueError,
                match="Surface mesh is not contiguous: 2 connected"):
        check_vol(vol)

def test_vol_nonmanifold():
    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 1]
    ])

    # Three tetrahedra share the same triangle (0, 1, 2)
    tets = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 4],
        [0, 1, 2, 5]
    ])

    vol = TetMesh(v=verts, t=tets)
    
    with pytest.raises(ValueError, match="Volume mesh is not manifold"):
        check_vol(vol)