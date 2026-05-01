from typing import Union
from warnings import warn
import numpy as np
from lapy import TriaMesh, Solver
from numpy.typing import ArrayLike

from neuromodes.eigen import EigenSolver
from neuromodes.basis import decompose, reconstruct

def truncate_emodes(geometry: TriaMesh, vfunc, threshold, 
                    evals=None, emodes=None, mass=None, threshold_method='decompose',
                    threshold_kwargs=None, output='group'): 

    # Prelims
    vf = np.asarray(vfunc, copy=True)
    vf = vf if vf.ndim > 1 else vf[:, np.newaxis] # ensure 2d for consistent processing
    n_maps = vf.shape[1]
    
    physical_threshold = threshold_method in ['eigenvalue', 'wavelength', 'fwhm']

    if threshold is None:
        if physical_threshold:
            thresholds = _convert_spatial_scale(estimate_fwhm(geometry, vf), input='fwhm', output=threshold_method)
        else: 
            raise ValueError(f"Threshold must be provided for threshold_method={threshold_method}.")
    elif np.isscalar(threshold):
        thresholds = np.full(n_maps, threshold)
    elif len(threshold) != n_maps:
        raise ValueError("Length of threshold array must match number of maps.")
    else: 
        thresholds = np.asarray(threshold)

    if threshold_kwargs is None: # have to set this outside the function declaration as {} is mutable
        threshold_kwargs = {}

    # Get data to threshold against
    if threshold_method == 'decompose':
        if emodes is None or mass is None:
            raise ValueError(f"emodes and mass must be provided when using threshold_method='{threshold_method}'.")
        betas = decompose(vf, emodes=emodes, mass=mass, **threshold_kwargs)
        ascending_data = np.cumsum(betas**2, axis=0) # user needs to demean if desired
        total_power = np.sum(vf * (mass @ vf), axis=0) # = np.diag(vf.T @ mass @ vf)
        thresholds = total_power * (1 - thresholds)
    elif threshold_method == 'reconstruct':
        if emodes is None or mass is None:
            raise ValueError(f"emodes and mass must be provided when using threshold_method='{threshold_method}'.")
        # TODO : change to new reconstruct
        # recons = reconstruct(data=vf, emodes=emodes, mass=mass)
        # errors = reconstruction_error(vf, recon=recons, mass=mass, **threshold_kwargs)
        _, errors, _ = reconstruct(data=vf, emodes=emodes, mass=mass, **threshold_kwargs)
        ascending_data = -errors
        thresholds = -thresholds
    elif threshold_method == 'eigenvalue':
        if evals is None:
            raise ValueError(f"evals must be provided when using threshold_method='{threshold_method}'.")
        ascending_data = np.broadcast_to(evals[:, np.newaxis], (len(evals), n_maps))
    elif threshold_method == 'fwhm' or threshold_method == 'wavelength': 
        if evals is None:
            raise ValueError(f"evals must be provided when using threshold_method='{threshold_method}'.")
        data = np.full(len(evals), np.inf) # default to inf for zero evals to avoid divide-by-zero issues
        data[1:] = convert_from_eigenvalue(evals[1:], output=threshold_method)
        ascending_data = -np.broadcast_to(data[:, np.newaxis], (len(evals), n_maps))
        thresholds = -thresholds
    else: 
        raise ValueError("Incorrect threshold_method specified.")

    # Find the required mode for each map
    # If the threshold is physical, keep only the modes that are strictly below the threshold
    # It the threshold is statistical, use the first mode that crosses the threshold 
    side = 'right' if physical_threshold else 'left' 
    n_mode = [np.searchsorted(ascending_data[:,i], thresholds[i], side=side) for i in range(n_maps)]
    n_mode = np.array(n_mode)
    if (failed_indices := np.where(n_mode == ascending_data.shape[0])[0]).size > 0:
        warn(f"Threshold not met for map(s) [{', '.join(map(str, failed_indices))}]. "
             f"All available modes were used. Consider providing more modes or loosening the threshold.")
    n_mode += not physical_threshold # this is the number of modes to use i.e. the first excluded mode

    # Return
    if output == 'mode': 
        result = n_mode
    elif output == 'group':
        result = mode_to_group(n_mode-1, method='ceil')+1 # number of groups (of last included mode)
    else: 
        raise ValueError("Output must be either 'mode' or 'group'.")
    
    return result if vfunc.ndim > 1 else result.item() # type: ignore # result will only be scalar if vfunc.ndim=1

def estimate_fwhm(surf: TriaMesh | Solver | EigenSolver, vfunc, method='fem', roi_mask=None):
    if method == 'wb':
        if isinstance(surf, TriaMesh):
            return _estimate_fwhm_wb(surf, vfunc, roi_mask=roi_mask)
        elif isinstance(surf, EigenSolver): 
            return _estimate_fwhm_wb(surf.geometry, vfunc, roi_mask=roi_mask) 
        elif isinstance(surf, Solver):
            raise TypeError("Solver instances cannot be used as they do not have a geometry attribute.")
        else:
            raise TypeError("For 'wb' method, geometry must be a TriaMesh or EigenSolver instance.")
        
    elif method == 'fem':
        if isinstance(surf, (Solver, EigenSolver)):
            stiffness, mass = surf.stiffness, surf.mass
        elif isinstance(surf, TriaMesh):
            stiffness, mass = Solver._fem_tria(surf) # doesn't matter if lumped or not
        else:
            raise TypeError("For 'fem' method, geometry must be a Solver/EigenSolver/TriaMesh instance.")
        return _estimate_fwhm_fem(stiffness, mass, vfunc, roi_mask=roi_mask)
    
    else:
        raise ValueError("Method must be either 'wb' or 'fem'.")

def _estimate_fwhm_wb(geometry: TriaMesh, vfunc, roi_mask=None):
    """Equivalent to wb_command -metric-estimate-fwhm"""
    # Prelims
    rois = np.ones(len(geometry.v), dtype=bool) if roi_mask is None else np.asarray(roi_mask, dtype=bool)
    vf = vfunc[rois, ...]

    # Get edges
    adj = geometry.adj_sym[rois, :][:, rois]
    rows, cols = adj.nonzero()
    rows, cols = rows[rows>cols], cols[rows>cols]  # Keep only upper triangle indices

    # Main computation
    vg = np.var(vf, axis=0, ddof=0)                           # global variance
    vl = np.mean((vf[rows, ...] - vf[cols, ...])**2, axis=0)  # local variance
    
    # In accordance with wb_command, use whole mesh mean edge length (not just mask)
    lambda_eff = 4 * -np.log(1 - vl / (2 * vg)) / geometry.avg_edge_length()**2
    return convert_from_eigenvalue(lambda_eff, output='fwhm') # alternate expression, but same as wb_command

def _estimate_fwhm_fem(stiffness, mass, vfunc, roi_mask=None):
    if roi_mask is not None: # subset stiffness, mass, and vfunc to roi (set diags to correct values)
        idx = np.asarray(roi_mask, dtype=bool)
        vf = vfunc[idx, ...]
        
        S = stiffness[idx, :][:, idx]
        S.setdiag(0)
        S.setdiag(-np.asarray(S.sum(axis=0)).ravel())

        M = mass[idx, :][:, idx]
        m_sum = np.asarray(M.sum(axis=0)).ravel()
        if M.nnz > M.shape[0]: # consistent
            target_mass = np.asarray(mass[:, idx].sum(axis=0)).ravel()
            M.setdiag(M.diagonal() + (target_mass - m_sum))
        else: # lumped
            M.setdiag(m_sum)

    else: 
        S, M, vf = stiffness, mass, vfunc

    # Vm = \Sigma_{i=1}^N \beta_i^2 (where \beta_i is the coefficient of mode i in the decomposition of vfunc)
    # Vs = \Sigma_{i=1}^N \beta_i^2 \lambda_i (where \lambda_i is the eigenvalue of mode i)
    # Vs/Vm = \lambda_{eff} where lambda is the effective eigenvalue of the map (weighted average)
    # If the input is a mode i, then this reduces to \lambda_i for that mode
    # TODO: change to demeanw / stats.py etc
    vf -= np.average(vf, weights=M.diagonal(), axis=0) # set (mass-weighted) mean to 0
    Vm = vf * (M @ vf)
    Vs = vf * (S @ vf) - 0.5 * (S @ vf**2) # keep second term for correction on open meshes
    lambda_eff = np.sum(Vs, axis=0) / np.sum(Vm, axis=0)
    return convert_from_eigenvalue(lambda_eff, output='fwhm')

def _estimate_fwhm_fem_local(stiffness, mass, vfunc):
    # TODO: change to demeanw / stats.py etc
    vf = vfunc - np.average(vfunc, weights=mass.diagonal(), axis=0) # set (mass-weighted) mean to 0
    Vm = vf * (mass @ vf)
    Vs = vf * (stiffness @ vf) - 0.5 * (stiffness @ vf**2)
    lambda_eff = Vs / Vm
    return convert_from_eigenvalue(lambda_eff, output='fwhm')
  
# TODO : clarify that for inputs of the form n^2-1, all methods return n-1 
# TODO : add overloads for int vs array inputs and int/float outputs
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
    opts = {
        'ceil':  (np.ceil, int),        # includes current group if mode_id is anywhere in it
        'floor': (np.floor, int),       # includes current group only if it is complete
        'round': (np.round, int),       # rounds to nearest group (if half, includes current group)
        'raw':   (lambda x: x, float)   # gives non-integer group index
    }

    # Setup
    if method not in opts:
        raise ValueError(f"Method must be one of {list(opts.keys())}")
    func, outtype = opts[method]
    mode_id_arr = np.asarray(mode_id)
    if not np.issubdtype(mode_id_arr.dtype, np.integer):
        warn("mode_id should be an integer or array of integers.")
    
    # Calcs
    result = func(np.sqrt(mode_id_arr + 1)).astype(outtype) - 1
    return result.item() if np.isscalar(mode_id) else result

# TODO : clarify that for inputs of the form n, all methods return (n+1)^2-1=n(n+2)
# TODO : add overloads for int vs array inputs and int/float outputs
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
    opts = {
        'ceil':  (np.ceil, int),        # modes up to and including all of the current group
        'floor': (np.floor, int),       # includes current group only if it is complete
        'round': (np.round, int),       # rounds to nearest complete group
        'raw':   (lambda x: x, float)   # gives (possibly) non-integer mode index
    }

    # Setup
    if method not in opts:
        raise ValueError(f"Method must be one of {list(opts.keys())}")
    func, outtype = opts[method]
    
    # Calcs
    result = func(np.asarray(group_id) + 1).astype(outtype)**2 - 1
    return result.item() if np.isscalar(group_id) else result

def convert_to_eigenvalue(value, input, area=None):
    if input == 'eigenvalue': 
        return value
    elif input == 'wavelength':
        return (2 * np.pi / value)**2
    elif input == 'fwhm':
        return 8 * np.log(2) / value**2
    elif input == 'group':
        if area is None:
            raise ValueError("Area must be provided when input is 'group'.")
        return value * (value + 1) * 4 * np.pi / area
    elif input == 'mode':
        if area is None:
            raise ValueError("Area must be provided when input is 'mode'.")
        return 4 * np.pi * value / area 
    else:
        raise ValueError("Incorrect input specified")

def convert_from_eigenvalue(eigenvalue, output, area=None):
    if output == 'eigenvalue': 
        return eigenvalue
    elif output == 'wavelength':
        return 2 * np.pi / np.sqrt(eigenvalue)
    elif output == 'fwhm':
        return np.sqrt(8 * np.log(2) / eigenvalue)
    elif output == 'group':
        if area is None:
            raise ValueError("Area must be provided when output is 'group'.")
        return (np.sqrt(eigenvalue * area / np.pi + 1) - 1) / 2
    elif output == 'mode':
        if area is None:
            raise ValueError("Area must be provided when output is 'mode'.")
        return eigenvalue * area / (4 * np.pi)
    else:
        raise ValueError("Incorrect output specified")
    
# TODO : consider making public
def _convert_spatial_scale(data, input, output, area=None):
    """Convenience function to convert between spatial scale representations without needing to manually convert to eigenvalues."""
    eigenvalue = convert_to_eigenvalue(data, input=input, area=area)
    return convert_from_eigenvalue(eigenvalue, output=output, area=area)
