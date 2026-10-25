from functools import partial

import numpy as np
import pytest
from scipy.sparse import csc_matrix, eye

from neuromodes.basis import decompose, calc_recon_error, reconstruct, filter_spectrum
from neuromodes.eigen import EigenSolver
from neuromodes.io import fetch_example_map, fetch_example_surf
from neuromodes.stats import sigmoid_rescale, zscorew


@pytest.fixture(scope='module')
def solver():
    surf, medmask = fetch_example_surf(res='4k')
    randmap = np.random.default_rng(0).standard_normal(size=medmask.sum())
    solver = EigenSolver(surf, mask=medmask)
    hetero = sigmoid_rescale(zscorew(randmap, solver.mass), steepness=0.5, upper=2.0)
    return solver.solve(n_modes=10, hetero=hetero)

def test_decompose_eigenmodes_1d(solver):
    for i in range(solver.n_modes):
        # Use an eigenmode as data
        coeffs = decompose(solver.emodes[:, i], solver.emodes, mass=solver.mass)

        # The mode should load onto only itself due to orthogonality
        coeffs_expected = np.zeros((solver.n_modes,))
        coeffs_expected[i] = 1
        assert np.allclose(coeffs, coeffs_expected, atol=1e-4), \
            f'Decomposition of mode {i} failed.'

def test_decompose_eigenmodes_2d(solver):
    emodes = solver.emodes
    coeffs = decompose(data=emodes, emodes=emodes, mass=solver.mass)
    coeffs_expected = np.eye(solver.n_modes)
    assert np.allclose(coeffs, coeffs_expected, atol=1e-4), \
        'Decomposition of modes onto themselves failed.'
    
def test_decompose_eigenmodes_3d(solver):
    emodes = solver.emodes
    data = np.stack((emodes, emodes), axis=2)
    coeffs = decompose(data=data, emodes=emodes, mass=solver.mass)
    coeffs_expected = np.stack((np.eye(solver.n_modes), np.eye(solver.n_modes)), axis=2)
    assert np.allclose(coeffs, coeffs_expected, atol=1e-4), \
        'Decomposition of modes onto themselves failed for 3D data.'

def test_decompose_invalid_data_shape(solver):

    with pytest.raises(ValueError, match=r"invalid shape"):
        decompose(np.ones(4002), solver.emodes, mass=solver.mass)

# def test_decompose_nan_data(solver):
#     data = np.ones(solver.n_verts)

#     bad_emodes = (solver.emodes).copy()

#     # TODO: move this to test_eigen.py, as it's covered by EigenData now
#     bad_emodes[0,0] = np.nan
#     with pytest.raises(ValueError, match="array must not contain infs or NaNs"):
#         decompose(data, bad_emodes, mass=solver.mass)

#     bad_emodes[0,0] = np.inf
#     with pytest.raises(ValueError, match="array must not contain infs or NaNs"):
#         decompose(data, bad_emodes, mass=solver.mass)

def test_decompose_massless(solver):

    with pytest.warns(UserWarning, match="Mass"):
        decompose(np.ones(solver.n_verts), solver.emodes)

def test_decompose_invalid_method(solver):

    with pytest.raises(ValueError,
                       match="Invalid method 'foo'; must be 'project' or 'regress'."):
        decompose(np.ones(solver.n_verts), solver.emodes, method='foo') # type: ignore

@pytest.fixture(scope='module')
def solver_32k():
    # Get modes of fsLR 32k midthickness (data is in 32k)
    mesh, medmask = fetch_example_surf()
    rng = np.random.default_rng(0)
    randmap = rng.standard_normal(size=medmask.sum())
    solver = EigenSolver(mesh, mask=medmask)
    hetero = sigmoid_rescale(zscorew(randmap, solver.mass), upper=2.0)
    solver.solve(10, hetero=hetero)
    return solver

# def test_decompose_nans(solver_32k):
#     # Decompose some maps
#     data = np.stack(
#         (fetch_example_map('fcgradient1')[solver_32k.mask],
#          fetch_example_map('myelinmap')[solver_32k.mask]),
#         axis=1
#     )

#     # turn checks off to simplify use of mass, which is irrelevant to this test
#     coeffs = decompose(data, solver_32k.emodes, method='regress',
#                        mass=csc_matrix(eye(solver_32k.n_verts)), checks='maps')

#     # Append data with NaNs (+100 vertices)
#     extraverts = 100
#     data_nans = np.concatenate([
#         data,
#         np.full((extraverts, data.shape[1]), np.nan)
#     ], axis=0)

#     # Add noise to modes and mass to match shapes (+100 vertices)
#     noise = np.random.default_rng().standard_normal((extraverts, solver_32k.n_modes))
#     modes_noise = np.concatenate([solver_32k.emodes, noise], axis=0)

#     # emodes/mass get masked according to the nans/s in data, leading to original coeffs values
#     with pytest.warns(UserWarning, match="values detected in data"):
#         coeffs_masked = decompose(data_nans, modes_noise, method='regress', checks='maps',
#                                   mass=csc_matrix(eye(solver_32k.n_verts+extraverts)))
#     assert np.allclose(coeffs, coeffs_masked), \
#         'coeffs values for project method are not close when data contains NaNs'

# TODO: more complicated version of above test, where three maps have two unique patterns of NaNs

# TODO: test that 'project' and 'regress' give very similar results across different mode_counts

@pytest.fixture(scope='module')
def gen_eigenmap(solver):

    # Use randomly weighted sums of modes to generate maps
    n_maps = 5
    weights = np.random.default_rng().standard_normal(size=(solver.n_modes, n_maps))
    eigenmaps = solver.emodes @ weights

    return eigenmaps, weights

class TestFilter:
    def test_filter_basic(self, solver, gen_eigenmap):
        eigenmaps, _ = gen_eigenmap

        filtered = filter_spectrum(eigenmaps, solver.emodes, solver.evals, mass=solver.mass)

        np.testing.assert_allclose(filtered, eigenmaps)

    def test_filter_multiple_mode_counts(self, solver, gen_eigenmap):
        eigenmaps, weights = gen_eigenmap

        mode_counts = range(1, solver.n_modes+1, 1)
        filtered = filter_spectrum(eigenmaps, solver.emodes, solver.evals, mass=solver.mass, mode_counts=mode_counts)

        # Check that the last filtered map matches the original map
        for i, mc in enumerate(mode_counts):
            np.testing.assert_allclose(filtered[:, :, i], solver.emodes[:, :mc] @ weights[:mc])

    def test_filter_multiple_mode_ids(self, solver, gen_eigenmap):
        eigenmaps, weights = gen_eigenmap

        mode_ids = [range(0, solver.n_modes//2), range(solver.n_modes//2, solver.n_modes)]
        filtered = filter_spectrum(eigenmaps, solver.emodes, solver.evals, mass=solver.mass, mode_ids=mode_ids)

        # Check that the filtered maps match the original maps for the specified mode IDs
        for i, ids in enumerate(mode_ids):
            np.testing.assert_allclose(filtered[:, :, i], solver.emodes[:, ids] @ weights[ids])

    def test_filter_exponential(self, solver, gen_eigenmap):
        eigenmaps, weights = gen_eigenmap

        # Apply an exponential filter to the eigenvalues
        filter_func = lambda evals: np.exp(-evals)
        filtered = filter_spectrum(eigenmaps, solver.emodes, solver.evals, mass=solver.mass, transfer_function=filter_func)

        # Check that the filtered maps are not equal to the original maps
        assert not np.allclose(filtered, eigenmaps), "Filtered maps should not be equal to the original maps."
        np.testing.assert_allclose(filtered, solver.emodes @ (filter_func(solver.evals)[:, np.newaxis] * weights))
        np.testing.assert_allclose(solver.decompose(filtered), filter_func(solver.evals)[:, np.newaxis] * weights)

    def test_filter_reciprocal(self, solver, gen_eigenmap):
        eigenmaps, weights = gen_eigenmap
        r = np.random.rand() + 1  # Regularization parameter, can't be (close to) 0 for this example

        # Apply a reciprocal filter to the eigenvalues (cf. GEM)
        filter_func = lambda evals: 1 / (1 + r**2 * evals)
        filtered = filter_spectrum(eigenmaps, solver.emodes, solver.evals, mass=solver.mass, transfer_function=filter_func)

        # Check that the filtered maps are not equal to the original maps
        assert not np.allclose(filtered, eigenmaps), "Filtered maps should not be equal to the original maps."
        np.testing.assert_allclose(filtered, solver.emodes @ (filter_func(solver.evals)[:, np.newaxis] * weights))
        np.testing.assert_allclose(solver.decompose(filtered), filter_func(solver.evals)[:, np.newaxis] * weights)

def test_reconstruct_project(solver):
    coeffs = decompose(solver.emodes, solver.emodes, mass=solver.mass)
    assert np.allclose(coeffs, np.eye(solver.n_modes), atol=1e-5), \
        'coeffs values do not match expected identity matrix when reconstructing modes onto themselves.'
    recon = reconstruct(coeffs, solver.emodes)
    assert np.allclose(recon, solver.emodes,
                       atol=1e-5), 'Final reconstructions do not match input modes.'
    
def test_reconstruct_regress_weighted(solver):
    coeffs = decompose(solver.emodes, solver.emodes, mass=solver.mass, method='regress')
    assert np.allclose(coeffs, np.eye(solver.n_modes), atol=1e-5), \
        'coeffs values do not match expected identity matrix when reconstructing modes onto themselves.'
    recon = reconstruct(coeffs, solver.emodes)
    assert np.allclose(recon, solver.emodes,
                       atol=1e-5), 'Final reconstructions do not match input modes.'

@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
def test_reconstruct_mode_superposition(solver, gen_eigenmap):
    eigenmaps, weights = gen_eigenmap

    coeffs = decompose(eigenmaps, solver.emodes, mass=solver.mass, mode_counts=range(1, solver.n_modes+1))
    # recon = reconstruct(coeffs, solver.emodes, mode_counts=range(1, solver.n_modes+1))
    # recon = np.stack(recon, axis=-1)
    recon = filter_spectrum(eigenmaps, solver.emodes, solver.evals, mass=solver.mass, mode_counts=range(1, solver.n_modes+1))

    # with pytest.warns(RuntimeWarning, match="invalid value encountered in divide"):
    correlation_error = calc_recon_error(eigenmaps, recon, metric='correlation', mass=solver.mass)
    euclidean_error = calc_recon_error(eigenmaps, recon, metric='euclidean', mass=solver.mass)

    # Correlation error should decrease from 1 to 0 when using mode 1 only versus all relevant modes
    assert np.allclose(recon[:,:,-1], eigenmaps,
                       atol=1e-5), 'Final reconstructions do not match input maps.'
    assert np.allclose(correlation_error[:,-1], 0,
                       atol=1e-5), 'Correlation error is not close to 0 when using all modes.'

    assert np.allclose(coeffs[-1], weights, atol=1e-4), \
        'coeffs values do not match input mode weights when using all modes.'

    # Euclidean error should be 0 when using all modes
    assert np.allclose(euclidean_error[:,-1], 0,
                       atol=1e-5), 'Euclidean error is not close to 0 when using all modes.'

    # Reconstruct using the first 5 modes, then the first 2 modes
    coeffs = decompose(eigenmaps, solver.emodes, mass=solver.mass, mode_counts=[5,2])
    recon = filter_spectrum(eigenmaps, solver.emodes, solver.evals, mass=solver.mass, mode_counts=[5,2])
    # recon = reconstruct(coeffs, solver.emodes, mode_counts=[5,2])
    correlation_error_modesq = calc_recon_error(eigenmaps, recon, mass=solver.mass)
    assert np.allclose(correlation_error_modesq[:,0], correlation_error[:,4]), \
        'Reconstruction scores do not match for 5 modes.'
    assert np.allclose(correlation_error_modesq[:,1], correlation_error[:,1]), \
        'Reconstruction scores do not match for 2 modes.'

def test_reconstruct_regress_method(solver, gen_eigenmap):
    eigenmaps, _ = gen_eigenmap

    coeffs = decompose(eigenmaps, emodes=solver.emodes, mass=csc_matrix(eye(solver.n_verts)), method='regress', mode_counts=range(1, solver.n_modes+1))
    # recon = reconstruct(coeffs, emodes=solver.emodes, mode_counts=range(1, solver.n_modes+1))
    recon = filter_spectrum(eigenmaps, solver.emodes, solver.evals, mass=csc_matrix(eye(solver.n_verts)), method='regress', mode_counts=range(1, solver.n_modes+1))
    correlation_error = calc_recon_error(eigenmaps, recon, metric='correlation', mass=csc_matrix(eye(solver.n_verts)))
    euclidean_error = calc_recon_error(eigenmaps, recon, metric='euclidean', mass=csc_matrix(eye(solver.n_verts)))

    # Errors should strictly decrease when adding modes
    assert np.all(np.diff(correlation_error[:, 1:], axis=1) < 0), \
        'Correlation error does not strictly decrease when adding modes.'  # nan error for constant mode (col 0), so ignore
    assert np.all(np.diff(euclidean_error, axis=1) < 0), \
        'Euclidean error does not strictly decrease when adding modes.'

def test_reconstruct_real_map(solver):
    map = fetch_example_map('myelinmap', res='4k')[solver.mask]
    coeffs = decompose(map, emodes=solver.emodes, mass=solver.mass, mode_counts=range(1, solver.n_modes+1))
    recon = reconstruct(coeffs, emodes=solver.emodes, mode_counts=range(1, solver.n_modes+1))
    recon = np.stack(recon, axis=-1)
    recon_error = calc_recon_error(map, recon, mass=solver.mass)

    # Correlation error should strictly decrease from 1, but not reach 0
    assert np.all(np.diff(recon_error[1:]) < 0), 'Reconstruction error does not strictly decrease.'
    assert not np.isclose(recon_error[-1], 0, atol=1e-6), \
        'Reconstruction error is unexpectedly close to 0 for only 10 modes.'

# def test_reconstruct_massless(solver):

#     coeffs = decompose(solver.emodes, solver.emodes, mass=solver.mass)
#     with pytest.warns(UserWarning, match="Mass"):
#         reconstruct(coeffs, solver.emodes)

class TestShape:

    @pytest.mark.parametrize("method", ['project', 'regress'])
    @pytest.mark.parametrize("data_shape", [(), (1,), (5,), (5,3)])
    def test_shape_none(self, solver, data_shape, method):
        data = np.random.default_rng().standard_normal((solver.n_verts,) + data_shape)

        coeffs = decompose(data, solver.emodes, mass=solver.mass, method=method)
        assert coeffs.shape == (solver.n_modes,) + data_shape, \
            'coeffs shape is incorrect for the given data shape.'
        
        recon = reconstruct(coeffs, solver.emodes)
        assert recon.shape == data.shape, \
            'recon shape is incorrect for the given data shape.'
        
        filtered = filter_spectrum(data, solver.emodes, solver.evals, mass=solver.mass, method=method)
        assert filtered.shape == data.shape, \
            'filtered shape is incorrect for the given data shape.'

    @pytest.mark.parametrize("method", ['project', 'regress'])
    @pytest.mark.parametrize("mode_counts", [1, 3, 5, 10, None])
    @pytest.mark.parametrize("data_shape", [(), (1,), (5,), (5,3)])
    def test_shape_mode_counts(self, solver, mode_counts, data_shape, method):
        data = np.random.default_rng().standard_normal((solver.n_verts,) + data_shape)

        coeffs = decompose(data, solver.emodes, mass=solver.mass, mode_counts=mode_counts)
        target_mode_counts = mode_counts if mode_counts is not None else solver.n_modes
        assert coeffs.shape == (target_mode_counts,) + data_shape, \
            'coeffs shape is incorrect for the given data shape.'

        recon = reconstruct(coeffs, solver.emodes, mode_counts=mode_counts)
        assert recon.shape == data.shape, \
            'recon shape is incorrect for the given data shape.'

        filtered = filter_spectrum(data, solver.emodes, solver.evals, mass=solver.mass, mode_counts=mode_counts)
        assert filtered.shape == data.shape, \
            'filtered shape is incorrect for the given data shape.'


    @pytest.mark.parametrize("method", ['project', 'regress'])
    @pytest.mark.parametrize("mode_counts", [[1,2,3], [5,10]])
    @pytest.mark.parametrize("data_shape", [(), (1,), (5,), (5,3)])
    def test_decompose_shape_mode_counts_list(self, solver, mode_counts, data_shape, method):
        data = np.random.default_rng().standard_normal((solver.n_verts,) + data_shape)

        coeffs = decompose(data, solver.emodes, mass=solver.mass, mode_counts=mode_counts)
        assert len(coeffs) == len(mode_counts), \
            'Number of coeffs outputs does not match number of mode counts.'
        for i in range(len(mode_counts)):
            assert coeffs[i].shape == (mode_counts[i],) + data_shape, \
                f'coeffs shape is incorrect for mode count {mode_counts[i]}'
            
        recon = reconstruct(coeffs, solver.emodes, mode_counts=mode_counts)
        assert len(recon) == len(mode_counts), \
            'Number of recon outputs does not match number of mode counts.'
        for i in range(len(mode_counts)):
            assert recon[i].shape == data.shape, \
                f'recon shape is incorrect for mode count {mode_counts[i]}'

        filtered = filter_spectrum(data, solver.emodes, solver.evals, mass=solver.mass, mode_counts=mode_counts)
        assert filtered.shape == data.shape + (len(mode_counts),), \
            'filtered shape is incorrect for the given data shape.'


    @pytest.mark.parametrize("method", ['project', 'regress'])
    @pytest.mark.parametrize("mode_ids", [[0,1,2,3,6,9], range(0,10,2)])
    @pytest.mark.parametrize("data_shape", [(), (1,), (5,), (5,3)])
    def test_decompose_shape_mode_ids(self, solver, mode_ids, data_shape, method):
        data = np.random.default_rng().standard_normal((solver.n_verts,) + data_shape)

        coeffs = decompose(data, solver.emodes, mass=solver.mass, mode_ids=mode_ids)
        assert coeffs.shape == (len(mode_ids),) + data_shape, \
            f'coeffs shape is incorrect for mode IDs {mode_ids}'

        recon = reconstruct(coeffs, solver.emodes, mode_ids=mode_ids)
        assert recon.shape == data.shape, \
            f'recon shape is incorrect for mode IDs {mode_ids}'

        filtered = filter_spectrum(data, solver.emodes, solver.evals, mass=solver.mass, mode_ids=mode_ids)
        assert filtered.shape == data.shape, \
            f'filtered shape is incorrect for mode IDs {mode_ids}'

    @pytest.mark.parametrize("method", ['project', 'regress'])
    @pytest.mark.parametrize("mode_ids", [[[1,2,3],[4,5,6],[7,8,9]],[range(0,10,2), range(0,10,3)]])
    @pytest.mark.parametrize("data_shape", [(), (1,), (5,), (5,3)])
    def test_decompose_shape_mode_ids_list(self, solver, mode_ids, data_shape, method):
        data = np.random.default_rng().standard_normal((solver.n_verts,) + data_shape)

        coeffs = decompose(data, solver.emodes, mass=solver.mass, mode_ids=mode_ids)
        assert len(coeffs) == len(mode_ids), \
            'Number of coeffs outputs does not match number of mode ID sets.'
        for i in range(len(mode_ids)):
            assert coeffs[i].shape == (len(mode_ids[i]),) + data_shape, \
                f'coeffs shape is incorrect for mode IDs {mode_ids[i]}'

        recon = reconstruct(coeffs, solver.emodes, mode_ids=mode_ids)
        assert len(recon) == len(mode_ids), \
            'Number of recon outputs does not match number of mode ID sets.'
        for i in range(len(mode_ids)):
            assert recon[i].shape == data.shape, \
                f'recon shape is incorrect for mode IDs {mode_ids[i]}'

        filtered = filter_spectrum(data, solver.emodes, solver.evals, mass=solver.mass, mode_ids=mode_ids)
        assert filtered.shape == data.shape + (len(mode_ids),), \
            'filtered shape is incorrect for the given data shape.'

