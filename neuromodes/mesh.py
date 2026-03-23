from typing import Union
import numpy as np
from lapy import TriaMesh, Solver
from numpy.typing import ArrayLike

from neuromodes.eigen import get_eigengroup_inds
from neuromodes.basis import decompose

def estimate_fwhm(geometry: TriaMesh, vfunc, roi_mask=None):
    """Equivalent to wb_command -metric-estimate-fwhm"""
    # Prelims
    rois = np.ones(len(geometry.v), dtype=bool) if roi_mask is None else np.asarray(roi_mask, dtype=bool)
    data = vfunc[rois, :] if vfunc.ndim > 1 else vfunc[rois, np.newaxis]

    # Get edges
    adj = geometry.adj_sym[rois, :][:, rois]
    rows, cols = adj.nonzero()
    rows, cols = rows[rows>cols], cols[rows>cols]  # Keep only upper triangle indices

    # Main computation
    vg = np.var(data, axis=0, ddof=0)                         # global variance
    vl = np.mean((data[rows, :] - data[cols, :])**2, axis=0)  # local variance
    # In accordance with wb_command, use whole mesh mean edge length (not just mask)
    fwhm = geometry.avg_edge_length() * np.sqrt(-2 * np.log(2) / np.log(1 - vl / (2 * vg)))

    return fwhm

def estimate_fwhm_fem(geometry: Solver, vfunc):
    vf = vfunc - np.average(vfunc, weights=geometry.mass.diagonal(), axis=0) # set (mass-weighted) mean to 0
    Vm = np.sum(vf * geometry.mass.dot(vf), axis=0)       # diag(vf.T @ M @ vf) 
    Vs = np.sum(vf * geometry.stiffness.dot(vf), axis=0)  # diag(vf.T @ S @ vf)
    lambda_eff = Vs / Vm
    fwhm = np.sqrt(8 * np.log(2) / lambda_eff)
    return fwhm
     
def truncate_emodes(geometry: TriaMesh, vfunc, emodes, evals, mass, per_spectrum=None): 
    # TODO : probably change this so that per_spectrum represents total accuracy (eg from reconstruct)
    # rather than the percentage of the spectrum
    # TODO : check that the number of modes is not just the max (error if j is at the end)
    # TODO : add option to return exact mode vs. complete group 
    if per_spectrum is not None:
        power = decompose(data=vfunc, emodes=emodes, mass=mass)**2
        power = power/np.sum(power)
        n_mode = np.searchsorted(np.cumsum(power), per_spectrum, side='right')
        n_group = mode_to_group(n_mode, method='ceil')
        return group_to_mode(n_group, method='ceil')
    else: 
        fwhm = estimate_fwhm(geometry, vfunc)
        with np.errstate(divide='ignore'):
            wavelengths = 2 * np.pi / np.sqrt(evals)
        n_mode = np.searchsorted(-wavelengths, -fwhm, side='right')
        n_group = mode_to_group(n_mode, method='round')
        return group_to_mode(n_group, method='ceil') # make it the whole group

# TODO : error if inputs are not integer
# TODO : clarify that for inputs of the form n^2-1, all methods return n-1 
def mode_to_group(mode_id: ArrayLike, method: str = 'ceil') -> Union[int, float, np.ndarray]:
    """
    Translates a linear mode index to its spherical harmonic group index.
    
    Parameters
    ----------
    mode_id : int or array_like
        The 0-indexed position of the eigenmode(s).
    method : {'round', 'floor', 'ceil', 'raw'}, optional
        How to resolve halfway points inside a degenerate group.
        'raw' returns the exact continuous fractional position.
    """
    # For method, get the (function, outputtype)
    ops = {
        'ceil':  (np.ceil, int),        # includes current group if mode_id is anywhere in it
        'floor': (np.floor, int),       # includes current group only if it is complete
        'round': (np.round, int),       # rounds to nearest group (if half, includes current group)
        'raw':   (lambda x: x, float)   # gives non-integer group index
    }

    # Setup
    if method not in ops:
        raise ValueError(f"Method must be one of {list(ops.keys())}")
    func, outtype = ops[method]
    
    # Calcs
    result = func(np.sqrt(np.asarray(mode_id) + 1)).astype(outtype) - 1 # have to use asarray for list inputs
    return result.item() if np.isscalar(mode_id) else result

# TODO : clarify that for inputs of the form n, all methods return (n+1)^2-1=n(n+2) 
def group_to_mode(group_id: ArrayLike, method: str = 'ceil') -> Union[int, float, np.ndarray]:
    """
    Translates a spherical harmonic group index back to a linear mode index.
    
    Parameters
    ----------
    group_id : int, float, or array_like
        The index of the harmonic group(s).
    method : {'ceil', 'floor', 'round', 'raw'}, optional
        How to resolve fractional modes. Default is 'ceil', which maps 
        an integer group ID to its terminal (final) mode index.
        'raw' mathematically inverts a fractional group back to its exact mode.
    """
    # For method, get the (function, outputtype)
    ops = {
        'ceil':  (np.ceil, int),        # modes up to and including all of the current group
        'floor': (np.floor, int),       # includes current group only if it is complete
        'round': (np.round, int),       # rounds to nearest complete group
        'raw':   (lambda x: x, float)   # gives (possibly) non-integer mode index
    }

    # Setup
    if method not in ops:
        raise ValueError(f"Method must be one of {list(ops.keys())}")
    func, outtype = ops[method]
    
    # Calcs
    result = func(np.asarray(group_id) + 1).astype(outtype)**2 - 1
    return result.item() if np.isscalar(group_id) else result

def wavelength_from_eigengroup_sphere(group_id, radius=1):
    return 2 * np.pi * radius / np.sqrt(group_id * (group_id + 1))

def eigenvalue_from_eigengroup_sphere(group_id, radius=1):
    return group_id * (group_id + 1) / radius**2

def fwhm_from_eigengroup_sphere(group_id, radius=1):
    return np.sqrt(8*np.log(2)) * radius / np.sqrt(group_id * (group_id + 1))

def wavelength_from_fwhm(fwhm):
    return np.pi / np.sqrt(2*np.log(2)) * fwhm

def wavelength_from_eigenvalue(evals):
    return 2 * np.pi / np.sqrt(evals)

def eigenvalue_from_surface_area(mode_id, area):
    return 4 * np.pi * mode_id / area
