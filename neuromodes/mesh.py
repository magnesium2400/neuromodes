from typing import Union
from warnings import warn
import numpy as np
from lapy import TriaMesh, Solver
from numpy.typing import ArrayLike

from neuromodes.eigen import EigenSolver
from neuromodes.basis import decompose, reconstruct

def estimate_fwhm(surf: TriaMesh | Solver | EigenSolver, vfunc, method='fem', roi_mask=None, output='fwhm'):
    if method == 'wb':
        if isinstance(surf, TriaMesh):
            return _estimate_fwhm_wb(surf, vfunc, roi_mask, output=output)
        elif isinstance(surf, EigenSolver): 
            return _estimate_fwhm_wb(surf.geometry, vfunc, roi_mask, output=output) 
        elif isinstance(surf, Solver):
            raise ValueError("Solver instances cannot be used as they do not have a geometry attribute.")
        else:
            raise ValueError("For 'wb' method, geometry must be a TriaMesh or EigenSolver instance.")
        
    elif method == 'fem':
        if roi_mask is not None:
            warn("roi_mask is not currently supported for 'fem' method and will be ignored.")

        if isinstance(surf, Solver) or isinstance(surf, EigenSolver):
            stiffness, mass = surf.stiffness, surf.mass
        elif isinstance(surf, TriaMesh):
            stiffness, mass = Solver._fem_tria(surf) # TODO : Check lump (true? false? either? default?)
        else:
            raise ValueError("For 'fem' method, geometry must be either a Solver instance or a TriaMesh instance.")
        return _estimate_fwhm_fem(mass, stiffness, vfunc, output=output)
    
    else:
        raise ValueError("Method must be either 'wb' or 'fem'.")

def _estimate_fwhm_wb(geometry: TriaMesh, vfunc, roi_mask=None, output='fwhm'):
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
    
    lambda_eff = convert_to_eigenvalue(fwhm, input_type='fwhm')
    return convert_from_eigenvalue(lambda_eff, output_type=output)

def _estimate_fwhm_fem(mass, stiffness, vfunc, output='fwhm'):
    # Vm = \Sigma_{i=1}^N \beta_i^2 (where \beta_i is the coefficient of mode i in the decomposition of vfunc)
    # Vs = \Sigma_{i=1}^N \beta_i^2 \lambda_i (where \lambda_i is the eigenvalue of mode i)
    # Vs/Vm = \lambda_{eff} where lambda is the effective eigenvalue of the map (weighted average)
    # If the input is an mode i, then this reduces to \lambda_i for that mode
    vf = vfunc - np.average(vfunc, weights=mass.diagonal(), axis=0) # set (mass-weighted) mean to 0
    Vm = vf * (mass @ vf)
    Vs = vf * (stiffness @ vf) - 0.5 * (stiffness @ vf**2)
    lambda_eff = np.sum(Vs, axis=0) / np.sum(Vm, axis=0)
    return convert_from_eigenvalue(lambda_eff, output_type=output)

def _estimate_fwhm_fem_local(mass, stiffness, vfunc, output='fwhm'):
    vf = vfunc - np.average(vfunc, weights=mass.diagonal(), axis=0) # set (mass-weighted) mean to 0
    Vm = vf * (mass @ vf)
    Vs = vf * (stiffness @ vf) - 0.5 * (stiffness @ vf**2)
    lambda_eff = Vs / Vm
    return convert_from_eigenvalue(lambda_eff, output_type=output)
     
def truncate_emodes(geometry: TriaMesh, vfunc, evals, emodes=None, mass=None, 
                    error_threshold=None, power_threshold=None, wavelength_threshold=None, 
                    reconstruct_kwargs={}, output='group'): 
    # TODO : probably change this so that per_spectrum represents total accuracy (eg from reconstruct)
    # rather than the percentage of the spectrum
    # TODO : check that the number of modes is not just the max (error if j is at the end)
    # TODO : add option to return exact mode vs. complete group 
    if error_threshold is not None:
        if emodes is None or mass is None:
            raise ValueError("emodes and mass must be provided when using error_threshold.")
        _, errors, _ = reconstruct(data=vfunc, emodes=emodes, mass=mass, **reconstruct_kwargs)
        errors = np.squeeze(errors)
        if error_threshold < errors[-1]: 
            raise RuntimeError(f"Error threshold {error_threshold} is too low. Consider providing more modes or increasing threshold.")
        n_mode = np.searchsorted(-errors, -error_threshold, side='right')
    elif power_threshold is not None:
        if emodes is None or mass is None:
            raise ValueError("emodes and mass must be provided when using power_threshold.")
        vf = vfunc - np.average(vfunc, weights=mass.diagonal(), axis=0)
        total_power = vf.T @ mass @ vf
        total_threshold = total_power * (1 - power_threshold)
        betas = decompose(vf, emodes=emodes, mass=mass)
        power = np.cumsum(betas**2)
        if power[-1] < total_threshold:
            raise RuntimeError(f"Power threshold {power_threshold} is too low. Consider providing more modes or increasing threshold.")
        n_mode = np.searchsorted(power, total_threshold, side='right')
    else: 
        if wavelength_threshold is None:
            wavelength_threshold = estimate_fwhm(geometry, vfunc, output='wavelength')
        wavelengths = convert_from_eigenvalue(evals, output_type='wavelength')
        if wavelength_threshold < wavelengths[-1]:
            raise RuntimeError(f"Wavelength threshold ({wavelength_threshold}) is smaller than the wavelength of the highest mode ({wavelengths[-1]}). Please provide more modes.")
        n_mode = np.searchsorted(-wavelengths, -wavelength_threshold, side='right')

    if output == 'mode': 
        return n_mode+1
    elif output == 'group':
        return mode_to_group(n_mode, method='ceil')+1
    else: 
        raise ValueError("Output must be either 'mode' or 'group'.")

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

def convert_to_eigenvalue(value, input_type='fwhm', area=None):
    if input_type == 'eigenvalue': 
        return value
    elif input_type == 'wavelength':
        return (2 * np.pi / value)**2
    elif input_type == 'fwhm':
        return 8 * np.log(2) / value**2
    elif input_type == 'group':
        if area is None:
            raise ValueError("Area must be provided when input_type is 'group'.")
        return value * (value + 1) * 4 * np.pi / area
    elif input_type == 'mode':
        if area is None:
            raise ValueError("Area must be provided when input_type is 'mode'.")
        return 4 * np.pi * value / area 
    else:
        raise ValueError("Incorrect input_type specified")

def convert_from_eigenvalue(eigenvalue, output_type='fwhm', area=None):
    if output_type == 'eigenvalue': 
        return eigenvalue
    elif output_type == 'wavelength':
        with np.errstate(divide='ignore'):
            return 2 * np.pi / np.sqrt(eigenvalue)
    elif output_type == 'fwhm':
        with np.errstate(divide='ignore'):
            return np.sqrt(8 * np.log(2) / eigenvalue)
    elif output_type == 'group':
        if area is None:
            raise ValueError("Area must be provided when output_type is 'group'.")
        return (np.sqrt(eigenvalue * area / np.pi + 1) - 1) / 2
    elif output_type == 'mode':
        if area is None:
            raise ValueError("Area must be provided when output_type is 'mode'.")
        return eigenvalue * area / (4 * np.pi)
    else:
        raise ValueError("Incorrect output_type specified")

