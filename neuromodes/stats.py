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

# TODO: ensure that all functions support nD input, not just 2D
def dotw(
    A: NDArray[floating],
    B: NDArray[floating],
    mass: csc_matrix | NDArray[floating],
    keepdims: bool = False
) -> NDArray[floating]:
    """Dot product between corresponding brain maps (not pairwise).
    """
    if A.shape != B.shape:
        raise ValueError(f"A and B must have matching shapes; got {A.shape} and {B.shape}.")

    mass = _process_vertex_areas(mass, A.shape[0])
    A_2d = A.reshape(A.shape[0], -1)
    B_2d = B.reshape(B.shape[0], -1)
    out = np.sum(A_2d * (mass @ B_2d), axis=0).reshape(A.shape[1:])
    return np.expand_dims(out, axis=0) if keepdims else out

def ssqw(
    A: NDArray[floating],
    mass: csc_matrix | NDArray[floating],
    keepdims: bool = False
) -> NDArray[floating]:
    """Energy (sum of squares) of each column."""
    return dotw(A, A, mass=mass, keepdims=keepdims)

def meanw(
    A: NDArray[floating],
    mass: csc_matrix | NDArray[floating],
    keepdims: bool = False
) -> float:
    """Area-weighted mean. Note that this is equivalent to the more general
    sum(mass @ A) / mass.sum()."""
    areas = _mass_to_areas(mass, A.shape[0])
    out = np.average(A, axis=0, weights=areas)
    return out[None, :] if keepdims else out  # TODO: test for A.ndim /= 2 cases

def demeanw(
    A: NDArray[floating],
    mass: csc_matrix | NDArray[floating]
) -> NDArray[floating]:
    """Remove the area-weighted mean."""
    return A - meanw(A, mass)

def varw(
    A: NDArray[floating],
    mass: csc_matrix | NDArray[floating],
    keepdims: bool = False
) -> float:
    """Area-weighted variance."""
    mass = _process_vertex_areas(mass, A.shape[0])
    B = demeanw(A, mass)
    out = ssqw(B, mass) / mass.sum()
    return out[None, :] if keepdims else out  # TODO: test for A.ndim /= 2 cases

def momentw(
    A: NDArray[floating],
    mass: csc_matrix | NDArray[floating],
    order: int,
    keepdims: bool = False
) -> float:
    """Area-weighted statistical moment of a given order."""
    if order == 1:
        return np.zeros(A.shape[1])
    elif order == 2:
        return varw(A, mass, keepdims=keepdims)
    else:
        # Approximate by lumping
        areas = _mass_to_areas(mass, A.shape[0])
        B = demeanw(A, areas)
        # Sum rows of the sparse matrix to get a lumped vector, safely flattened
        return meanw(B ** order, areas, keepdims=keepdims)

def stdw(
    A: NDArray[floating],
    mass: csc_matrix | NDArray[floating],
    keepdims: bool = False
) -> float:
    """Area-weighted standard deviation."""
    return np.sqrt(varw(A, mass, keepdims=keepdims))

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
    mass: csc_matrix | NDArray[floating]
) -> NDArray[floating]:
    """
    Weighted covariance.
    Usage: covw(A, mass=w) computes covariance of A with itself.
           covw(A, B, mass=w) computes cross-covariance between A and B.
    No bias correction is applied, equivalent to np.cov(..., bias=True). Since mesh vertices are not
    IID samples and maps typically display spatial autocorrelation, Bessel's N-1 correction is not
    appropriate.
    """
    mass = _process_vertex_areas(mass, A.shape[0])
    areas = _mass_to_areas(mass, A.shape[0])
    total_area = areas.sum()
    
    A_d = demeanw(A, areas)
    B_d = demeanw(B, areas) if B is not None else A_d
    gram = gramw(A_d, B_d, mass=mass)
    
    return gram / total_area

def vecnormw(
    A: NDArray[floating],
    mass: csc_matrix | NDArray[floating],
    p: int = 2,
    keepdims: bool = False
) -> NDArray[floating]:
    """Calculates the area-weighted L^p norm of spatial maps."""
    if p == 2: # Exact (well-defined)
        return np.sqrt(ssqw(A, mass, keepdims=keepdims))
    
    elif p == np.inf: # Exact (ignores the mass matrix)
        out = np.max(np.abs(A), axis=0)
        return out[None, :] if keepdims else out
    
    else: # Approximate by lumping
        areas = _mass_to_areas(mass, A.shape[0])
        out = np.sum(areas * (np.abs(A) ** p), axis=0) ** (1 / p)
        return out[None, :] if keepdims else out

def cdistw(
    X: NDArray[floating],
    Y: NDArray[floating],
    mass: csc_matrix | NDArray[floating],
    metric: _MetricCallback | _MetricKind = 'euclidean'
) -> NDArray[floating]:
    """Pairwise distance between rows of X and Y, accounting for mass matrix. Some functions support
    exact calculation (these have been reimplemented); some functions are approximated by lumping
    (scipy.spatial.distance.cdist with weights)."""
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
            raise ValueError(f"Mass matrix has invalid shape: {mass_arr.shape} (should be square "
                             "or vector).")

    elif isinstance(mass, csc_matrix) and mass.shape is not None: # appease: pyright
        if mass.shape[0] == mass.shape[1]: 
            output = mass
        elif mass.shape[0] == 1 or mass.shape[1] == 1:
            output = diags(mass.toarray().flatten(), format='csc')
        else: 
            raise ValueError(f"Sparse mass matrix has invalid shape: {mass.shape} (should be "
                             "square or vector).")
            
    else:
        raise TypeError("mass must be a 1D array of vertex areas, a 2D mass matrix, or a sparse "
                        "matrix.")

    output = csc_matrix(output)
    
    if output.shape is None: 
        raise ValueError("Mass matrix has undefined shape.")
    elif output.shape[0] != output.shape[1]:
        raise ValueError(f"Mass matrix must be square; got shape {output.shape}.")
    if n_verts is not None and output.shape != (n_verts, n_verts):
        raise ValueError(f"Mass matrix has invalid shape: {output.shape} (should be ({n_verts}, "
                         f"{n_verts})).")

    return output