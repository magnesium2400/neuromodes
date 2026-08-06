import numpy as np
import pytest
from neuromodes.eigen import EigenSolver
from neuromodes.io import fetch_surf
from neuromodes.mbm2 import mbm_example_workflow

def generate_mbm_test_data(
        emodes: np.ndarray, 
        n_subjects: int = 50, 
        n_sig_modes: int = 5, 
        effect_size: float = 3.0,
        seed: int | None = 42
):
    """
    Generates synthetic vertex maps with known significant eigenmodes 
    for a two-sample group comparison.
    """
    rng = np.random.default_rng(seed)
    n_vertices, n_modes = emodes.shape
    
    # Randomly select which modes will be our ground-truth significant modes
    sig_modes = rng.choice(n_modes, size=n_sig_modes, replace=False)
    sig_modes.sort()
    
    # Create the two-sample design matrix
    group_A_idx = np.arange(n_subjects // 2)
    group_B_idx = np.arange(n_subjects // 2, n_subjects)
    
    statDesignMatrix = np.zeros((n_subjects, 2), dtype=bool)
    statDesignMatrix[group_A_idx, 0] = True
    statDesignMatrix[group_B_idx, 1] = True

    # Generate mode loadings
    mode_loadings = rng.normal(loc=0.0, scale=1.0, size=(n_modes, n_subjects))
    
    # Inject the signal into the chosen significant modes
    mode_loadings[sig_modes[:, None], group_A_idx] = rng.normal(
        loc=+effect_size, scale=1.0, size=(len(sig_modes), len(group_A_idx))
    )
    mode_loadings[sig_modes[:, None], group_B_idx] = rng.normal(
        loc=-effect_size, scale=1.0, size=(len(sig_modes), len(group_B_idx))
    )
        
    # Project the mode loadings back to vertex space
    maps = emodes @ mode_loadings
    
    # Add a layer of vertex-wise spatial noise
    vertex_noise = rng.normal(loc=0.0, scale=0.5, size=(n_vertices, n_subjects))
    maps += vertex_noise

    return maps, statDesignMatrix, sig_modes

def test_mbm_example_workflow():
    # Setup surface and eigenmodes
    mesh, _ = fetch_surf(density='4k') 
    solver = EigenSolver(mesh).solve(100)

    # Generate synthetic data
    maps, statDesignMatrix, ground_truth_sig_modes = generate_mbm_test_data(
        solver.emodes, 
        n_subjects=50, 
        n_sig_modes=5,
        effect_size=3.0, 
        seed=314
    )

    # Run the workflow
    reconMap, sig_mode_indices = mbm_example_workflow(
        maps=maps,
        emodes=solver.emodes,
        mass=solver.mass,
        statTest='two sample',
        statDesignMatrix=statDesignMatrix,
        statPThr=0.05,
        statFDR=True,
        n_modes=solver.emodes.shape[1],
        n_permutations=200,
        seed=271828
    )

    # Assert that all ground truth significant modes were identified by the workflow
    assert np.all(np.isin(ground_truth_sig_modes, sig_mode_indices)), \
        f"Expected modes {ground_truth_sig_modes} but found {sig_mode_indices}"
    np.testing.assert_array_equal(
        sig_mode_indices, 
        ground_truth_sig_modes, 
    )
    