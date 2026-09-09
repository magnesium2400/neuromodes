"""
Module for generating models of cortical structural connectomes.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from neuromodes.eigen import EigenData

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from neuromodes.eigen import _CheckKind
    from scipy.sparse import spmatrix

def compute_gem(
    emodes: NDArray[np.floating],
    evals: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating] | None = None,
    r: float = 9.53,
    k: int = 108,
    euclidean_pinv: bool = False,
    checks: _CheckKind = 'shape'
) -> NDArray[np.floating]:
    """
    Generate a model structural connectome using the Geometric Eigenmode Model [1]_.

    Parameters
    ----------
    emodes : array-like
        The eigenmodes array of shape ``(n_verts, n_modes)``, where ``n_verts`` is the number of
        vertices and ``n_modes`` is the number of eigenmodes.
    evals : array-like
        The eigenvalues array of shape ``(n_modes,)``.
    r : float, optional
        Spatial scale parameter for the Green's function, in millimeters. Default is ``9.53``.
    k : int, optional
        Number of eigenmodes to use. Default is ``108``.
    mass : spmatrix | NDArray[np.floating] | None, optional
        The mass matrix of shape ``(n_verts, n_verts)``. If ``euclidean_pinv`` is ``True``, this
        parameter is ignored. If ``None``, the identity matrix is used. Default is ``None``.
    euclidean_pinv : bool, optional
        Whether to use the Euclidean pseudo-inverse of the eigenmodes (equivalent to ``(emodes[:,
        :k].T @ emodes[:, :k])**(-1) @ emodes[:, :k].T``). This matches the original implementation
        [1]_ but does not account for mesh irregularities. If ``False``, uses the mass-weighted
        pseudo-inverse (``emodes[:, :k].T @ mass``). Default is ``False``.
    checks : bool, optional
        Whether to validate types and shapes of ``emodes``, ``evals``, and ``mass`` before
        computation. Default is ``True``.

    Returns
    -------
    np.ndarray
        The generated structural connectivity matrix of shape ``(n_verts, n_verts)``.

    Raises
    ------
    ValueError
        If ``r`` is not a positive number.
    ValueError
        If ``k`` is not a positive integer in the range [1, ``n_modes``].

    Notes
    -----
    If comparing this model to empirical connectomes, it is recommended to threshold the returned
    matrix to match the density of the empirical data.

    Prior work has treated ``r`` and ``k`` as free parameters to fit empirical data [1]_, with the
    default value reflecting an optimal fit to human diffusion MRI data. Consider adjusting these
    parameters, as their optima can vary across analyses (e.g., different surfaces, heterogeneous
    modes, parcellated connectomes, empirical data, fitting metrics, etc.).
    
    While the model can be computed using non-cortical modes, users should consider whether this is
    theoretically sensible and physiologically plausible.

    References
    ----------
    ..  [1] Normand, F., et al. (2025). Geometric constraints on the architecture of mammalian
        cortical connectomes. BioRxiv. https://doi.org/10.1101/2025.09.17.676944
    """
    # Format / validate arguments
    if checks is not False:
        ved = EigenData(emodes=emodes, evals=evals, mass=mass, checks=checks)
        emodes, evals, mass = ved.emodes, ved.evals, ved.mass

    r = float(r)
    n_modes = emodes.shape[1]
    if r <= 0:
        raise ValueError("Parameter r must be a positive number.")
    if k != int(k) or k <= 0 or k > n_modes:
        raise ValueError(f"Parameter k must be an integer in the range [1, n_modes = {n_modes}].")

    # Compute pseudoinverse of eigenmodes
    if euclidean_pinv:
        pinv = np.linalg.pinv(emodes[:, :k])
    elif mass is None:
        pinv = emodes[:, :k].T
    else:
        pinv = emodes[:, :k].T @ mass

    # Construct model
    denom = 1/(1 + evals[:k] * r**2)
    gem = emodes[:, :k] @ (denom[:, None] * pinv)

    # Replace diagonal and negative values with zero
    np.fill_diagonal(gem, 0)
    gem = np.maximum(gem, 0)

    # Symmetrise (only removes floating point error if euclidean_pinv is False)
    gem = (gem + gem.T) / 2

    # Normalise
    return gem / np.max(gem)