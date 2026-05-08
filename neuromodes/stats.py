"""
Mass (or simply area)-weighted adaptations of common statistical functions for spatial maps.
Conventional functions are equivalent to setting mass to identity, representing a mesh where each
vertex has Voronoi area/volume of 1.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING
from warnings import warn
from scipy.spatial.distance import squareform, cdist
from scipy.stats import rankdata
from scipy.sparse import csc_matrix, diags

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray
    from scipy.spatial.distance import _MetricCallback, _MetricKind

def gramw(
    A: NDArray[floating],
    B: NDArray[floating] | None = None,
    *,
    mass: csc_matrix | NDArray[floating]
) -> NDArray[floating]:
    """Dot product between all columns (pairwise)."""
    mass = _process_vertex_areas(mass, A.shape[0])
    if B is None:
        B = A
    return A.T @ (mass @ B)

# TODO : consider adding keepdims parameter to many of theses funcs
# TODO : consider changing to dotw(A, mass), where A can be A = [A_1, A_2]
def dotw(
    A: NDArray[floating],
    B: NDArray[floating],
    mass: csc_matrix | NDArray[floating]
) -> NDArray[floating]:
    """Dot product between corresponding columns (not pairwise)."""
    mass = _process_vertex_areas(mass, A.shape[0])
    return np.sum(A * (mass @ B), axis=0)

def ssqw(
    A: NDArray[floating],
    mass: csc_matrix | NDArray[floating]
) -> NDArray[floating]:
    """Energy (sum of squares) of each column."""
    return dotw(A, A, mass=mass)

def meanw(
    A: NDArray[floating],
    mass: csc_matrix | NDArray[floating]
) -> float:
    """Area-weighted mean."""
    mass = _process_vertex_areas(mass, A.shape[0])
    areas = np.asarray(mass.sum(axis=1))
    return (areas * A).sum(axis=0) / areas.sum()

def demeanw(
    A: NDArray[floating],
    mass: csc_matrix | NDArray[floating]
) -> NDArray[floating]:
    """Remove the area-weighted mean."""
    return A - meanw(A, mass)

def varw(
    A: NDArray[floating],
    mass: csc_matrix | NDArray[floating]
) -> float:
    """Area-weighted variance."""
    mass = _process_vertex_areas(mass, A.shape[0])
    B = demeanw(A, mass)
    return ssqw(B, mass) / mass.sum()

def momentw(
    A: NDArray[floating],
    mass: csc_matrix | NDArray[floating],
    order: int
) -> float:
    """Area-weighted statistical moment of a given order."""
    if order == 1:
        return np.zeros(A.shape[1])
    elif order == 2:
        return varw(A, mass)
    else:
        # Approximate by lumping
        mass = _process_vertex_areas(mass, A.shape[0])
        B = demeanw(A, mass)
        # Sum rows of the sparse matrix to get a lumped vector, safely flattened
        areas = np.asarray(mass.sum(axis=1))
        return np.sum(areas * (B ** order)) / areas.sum()

def stdw(
    A: NDArray[floating],
    mass: csc_matrix | NDArray[floating]
) -> float:
    """Area-weighted standard deviation."""
    return np.sqrt(varw(A, mass))

def zscorew(
    A: NDArray[floating],
    mass: csc_matrix | NDArray[floating]
) -> NDArray[floating]:
    """Z-score using area-weighted mean and standard deviation."""
    return demeanw(A, mass) / stdw(A, mass)

def covw(
    A: NDArray[floating],
    B: NDArray[floating] | None = None,
    *,
    mass: csc_matrix | NDArray[floating],
    bias: bool = False
) -> NDArray[floating]:
    """
    Weighted covariance.
    Usage: covw(A, mass=w) computes covariance of A with itself.
           covw(A, B, mass=w) computes cross-covariance between A and B.
    If bias is False, apply the standard weighted Bessel correction for diagonal mass matrices.
    """
    mass = _process_vertex_areas(mass, A.shape[0])
    areas = np.asarray(mass.sum(axis=1)).ravel()
    total_area = areas.sum()
    
    A_d = demeanw(A, mass)
    if B is None:
        gram = gramw(A_d, mass=mass)
    else:
        B_d = demeanw(B, mass)
        gram = gramw(A_d, B_d, mass=mass)

    norm = total_area
    if not bias:
        norm -= (areas @ areas) / total_area
    return gram / norm

def vecnormw(
    A: NDArray[floating],
    mass: csc_matrix | NDArray[floating],
    p: int = 2
) -> NDArray[floating]:
    """Calculates the area-weighted L^p norm of spatial maps."""
    if p == 2: # Exact (well-defined)
        return np.sqrt(ssqw(A, mass))
    
    elif p == np.inf: # Exact (ignores the mass matrix)
        return np.max(np.abs(A), axis=0)
    
    else: # Approximate by lumping
        mass = _process_vertex_areas(mass, A.shape[0])
        areas = np.asarray(mass.sum(axis=1))
        return np.sum(areas * (np.abs(A) ** p), axis=0) ** (1 / p)

def cdistw(
    X: NDArray[floating],
    Y: NDArray[floating],
    mass: csc_matrix | NDArray[floating],
    metric: _MetricCallback | _MetricKind = 'euclidean'
) -> NDArray[floating]:
    """Pairwise distance between rows of X and Y, accounting for mass matrix. Some functions support
    exact calculation (these have been reimplemented); some functions are approximated by lumping
    (scipy.spatial.distance.cdist with weights)."""
    
    mass = _process_vertex_areas(mass, X.shape[0])
    
    # Reimplement these to support unlumped mass matrix
    if metric == 'sqeuclidean':
        D = ssqw(X, mass)[:, None] + ssqw(Y, mass)[None, :] - 2 * gramw(X, Y, mass=mass)
        
    elif metric == 'euclidean':
        D = np.sqrt(cdistw(X, Y, mass, 'sqeuclidean'))
        
    elif metric == 'cosine':
        Num = gramw(X, Y, mass=mass)
        DenX = vecnormw(X, mass, 2)[:, None]
        DenY = vecnormw(Y, mass, 2)[None, :]
        D = 1 - Num / (DenX * DenY)
        
    elif metric == 'correlation':
        D = cdistw(demeanw(X, mass), demeanw(Y, mass), mass, 'cosine')
        
    else:
        areas = _mass_to_areas(mass, X.shape[0])
        D = cdist(X, Y, metric=metric, mass=areas)

    return np.maximum(D, 0)

def pdistw(
    X: NDArray[floating],
    mass: csc_matrix | NDArray[floating],
    metric: _MetricCallback | _MetricKind = 'euclidean'
) -> NDArray[floating]:
    """Pairwise distances between observations in X, outputting a condensed vector."""
    D2 = cdistw(X, X, mass, metric)
    np.fill_diagonal(D2, 0) # Ensures exact 0 on diagonal
    return squareform(D2, checks=False)

def correlationw(
    XA: NDArray[floating],
    XB: NDArray[floating],
    mass: csc_matrix | NDArray[floating],
    metric: str = 'pearsonr'
) -> NDArray[floating]:    
    if metric=='spearmanr':
        XA = rankdata(XA, axis=0)
        XB = rankdata(XB, axis=0)
    elif metric != 'pearsonr': 
        raise ValueError(f"Invalid metric '{metric}'; must be 'pearsonr' or 'spearmanr'.")
    return 1 - cdistw(XA, XB, mass=mass, metric='correlation')

def solvew(
    A: NDArray[floating],
    B: NDArray[floating],
    mass: csc_matrix | NDArray[floating]
) -> NDArray[floating]:
    """
    Use method of normal equations to give area-weighted least squares error.
    See https://en.wikipedia.org/wiki/Weighted_least_squares#Motivation
    """
    mass = _process_vertex_areas(mass, A.shape[0])
    # Solves (A'WA)x = (A'WB)
    return np.linalg.solve(A.T @ mass @ A, A.T @ mass @ B)

def lstsqw(
    a: NDArray[floating],
    b: NDArray[floating],
    mass: csc_matrix | NDArray[floating],
    rcond: float | None = None
) -> tuple[NDArray[floating], int, float, NDArray[floating]]:
    """
    Solve the weighted least squares by lumping the mass matrix and weighting each vertex.
    """
    va = np.sqrt(_mass_to_areas(mass, a.shape[0])) # (n_verts,)
    aw = a * va[:, np.newaxis]
    bw = b * va[:, np.newaxis] if b.ndim != 1 else b * va
    return np.linalg.lstsq(aw, bw, rcond=rcond)

def _mass_to_areas(
    mass: csc_matrix | NDArray[floating] | None = None,
    n_verts: int | None = None
) -> NDArray[floating]:
    mass = _process_vertex_areas(mass=mass, n_verts=n_verts)
    return np.asarray(mass.sum(axis=0)).ravel()

# TODO: consider adding dtype parameter to _process_vertex_areas for w=None case
def _process_vertex_areas(
    mass: csc_matrix | NDArray[floating] | None = None,
    n_verts: int | None = None
) -> csc_matrix:
    
    if mass is None and n_verts is None:
        raise ValueError("Either mass or n_verts must be provided.")
    
    elif mass is None and n_verts is not None: # appease: pyright
        warn("Mass matrix not provided; assuming that area at each vertex is 1")
        output = diags(np.ones(n_verts), format='csc')

    elif isinstance(mass, (np.ndarray, list)):        
        mass_arr = np.asarray(mass)
        if mass_arr.ndim == 2 and mass_arr.shape[0] == mass_arr.shape[1]:
            output = mass_arr
        elif mass_arr.ndim == 1 or mass_arr.shape[0] == 1 or mass_arr.shape[1] == 1:
            output = diags(np.ravel(mass_arr), format='csc')
        else: 
            raise ValueError(f"Mass matrix has invalid shape: {mass_arr.shape} (should be square or vector).")

    elif isinstance(mass, csc_matrix) and mass.shape is not None: # appease: pyright
        if mass.shape[0] == mass.shape[1]: 
            output = mass
        elif mass.shape[0] == 1 or mass.shape[1] == 1:
            output = diags(mass.toarray().flatten(), format='csc')
        else: 
            raise ValueError(f"Sparse mass matrix has invalid shape: {mass.shape} (should be square or vector).")
            
    else:
        raise TypeError("mass must be a 1D array of vertex areas, a 2D mass matrix, or a sparse matrix.")

    output = csc_matrix(output)
    
    if output.shape is None: 
        raise ValueError("Mass matrix has undefined shape.")
    elif output.shape[0] != output.shape[1]:
        raise ValueError(f"Mass matrix must be square; got shape {output.shape}.")
    if n_verts is not None and output.shape != (n_verts, n_verts):
        raise ValueError(f"Mass matrix has invalid shape: {output.shape} (should be ({n_verts}, {n_verts})).")

    return output