"""
Module for expressing brain maps as linear combinations of orthogonal basis vectors, such as
geometric eigenmodes.
"""

from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING
from warnings import warn
import numpy as np
from scipy.spatial.distance import cdist
from neuromodes.eigen import EigenData
from neuromodes.mesh import mask_laplacian

if TYPE_CHECKING:   
    from numpy import floating
    from numpy.typing import NDArray, ArrayLike
    from scipy.spatial.distance import _MetricCallback, _MetricKind 
    from scipy.sparse import csc_matrix

nan_warning = ("data contains NaNs and/or Infs; these will be disregarded during decomposition by "
               "masking corresponding vertices from data and emodes.")

def decompose(
    data,
    emodes,
    method = 'project',
    mass = None,
    mode_counts = None,
    mode_ids = None,
    checks = True,
):
    """
    Calculate the decomposition of the given data onto a basis set.

    Parameters
    ----------
    data : array-like
        The input data array of shape ``(n_verts, ...)``, where ``n_verts`` is the number of
        vertices and ``n_maps`` is the number of maps.
    emodes : array-like
        The vectors array of shape ``(n_verts, n_modes)``, where ``n_modes`` is the number of basis
        vectors.
    method : str, optional
        The method used for the decomposition, either ``'project'`` to project data into a
        mass-orthonormal space or ``'regress'`` for least-squares fitting. Note that the beta values
        from ``'regress'`` tend towards those from ``'project'`` when more basis vectors are
        provided. For a non-orthonormal basis set, ``'regress'`` must be used. Default is
        ``'project'``.
    mass : array-like, optional
        The mass matrix of shape ``(n_verts, n_verts)`` used for the decomposition when method is
        ``'project'``. If vectors are orthonormal in Euclidean space, leave as ``None``. See
        :func:`eigen.is_orthonormal_basis` for more details. Default is ``None``.
    checks : str or bool, optional
        Whether to verify types, shapes, and orthonormality of ``emodes`` and ``mass`` before
        decomposition. Default is ``True``.

    Returns
    -------
    numpy.ndarray or List
        The beta coefficients array of shape ``(n_modes, ...)`` or ``List of (n_modes, ...)``
        arrays, obtained from the decomposition.
    
    Raises
    ------
    ValueError
        If ``data`` does not have shape ``(n_verts,)`` or ``(n_verts, n_maps)``.
    ValueError
        If ``method`` is not ``'project'`` or ``'regress'``.

    Notes
    -----
    If ``data`` contains NaNs or Infs, these will be disregarded during decomposition by masking
    corresponding vertices from ``data``, ``emodes``, and ``mass``. Note that this can lead to
    unexpected behaviour, such as extreme values in affected areas of the reconstructed data, or
    extreme beta values. This appears particularly prevalent when using the ``'regress'`` method. 
    """
    # Format / validate inputs
    if method not in ['project', 'regress']:
        raise ValueError(f"Invalid method '{method}'; must be 'project' or 'regress'.")
    
    # Skip warning for EigenSolver, where mass is always passed in
    if (method == 'regress') and (mass is not None) and (checks is True or checks == 'ortho'):  
            warn("mass is ignored when method='regress'.")
    
    if checks is not False: 
        ved = EigenData(emodes=emodes, mass=mass, data=data, checks=checks)
        emodes, mass, data = ved.emodes, ved.mass, ved.data

    mode_ids, squeeze_output = _process_mode_ids(mode_counts, mode_ids, emodes.shape[1])
    n_modes = [len(x) for x in mode_ids]

    # Manipulate input/output shapes
    output_shapes = [(i,) + data.shape[1:] for i in n_modes]
    output = [np.empty(shape) for shape in output_shapes]

    # TODO : only need to decompose once (with n=max modes) if using orthogonal method
    # Handle NaNs and Infs by masking out afflicted vertices (separately for each NaN/Inf pattern)
    data_reshaped = data.reshape(data.shape[0], -1) # guaranteed 2d
    data_finite = np.isfinite(data_reshaped)
    masks, mask_indices = np.unique(data_finite, axis=1, return_inverse=True)
    for j in range(len(mode_ids)):
        tmp = np.empty((n_modes[j], data_reshaped.shape[1]))
        for i, mask in enumerate(masks.T):
            # Get indices of maps with this NaN/Inf pattern
            # Remove verts with NaNs/Inf in this group from data and emodes
            # Calculate beta coefficients for subset of data
            map_indices = np.where(mask_indices == i)[0]
            tmp[:, map_indices] = _calc_beta(
                data = data_reshaped[:, map_indices], 
                emodes = emodes[:, mode_ids[j]], 
                method = method,
                mass = mass, 
                mask = mask
            )
        output[j] = tmp.reshape(output_shapes[j])

    if squeeze_output:
        output = output[0]
    return output

def reconstruct(
    data: NDArray,
    emodes: NDArray,
    method: str = 'project',
    mass: csc_matrix | None = None,
    mode_counts: List | Tuple | None = None,
    mode_ids: List | Tuple | None = None,
    metric: _MetricCallback | _MetricKind | None = 'correlation',
    checks: bool | str | None = None,
    **cdist_kwargs
) -> Tuple[NDArray[floating], NDArray[floating], list[NDArray[floating]]]:
    """
    Calculate and score the reconstruction of the given independent data using the provided
    orthogonal vectors (e.g., geometric eigenmodes).

    Parameters
    ----------
    data : array-like
        The input data array of shape ``(n_verts, ...)``, where ``n_verts`` is
        the number of vertices and ``n_maps`` is the number of brain maps.
    emodes : array-like
        The vectors array of shape ``(n_verts, n_modes)``, where ``n_modes`` is the number of
        orthogonal vectors.
    method : str, optional
        The method used for the decomposition, either ``'project'`` to project data into a
        mass-orthonormal space or ``'regress'`` for least-squares fitting. Note that the beta values
        from ``'regress'`` tend towards those from ``'project'`` when more basis vectors are
        provided. For a non-orthonormal basis set, ``'regress'`` must be used. Default is
        ``'project'``.
    mass : array-like, optional
        The mass matrix of shape ``(n_verts, n_verts)`` used for the decomposition when method is
        ``'project'``. If vectors are orthonormal in Euclidean space, leave as ``None``. See
        :func:`eigen.is_orthonormal_basis` for more details. Default is ``None``.
    mode_counts : array-like, optional
        The sequence of vectors to be used for reconstruction, of shape ``(n_recons,)``. For
        example, ``mode_counts=np.array([10,20,30])`` will run three analyses: with the first 10
        vectors, with the first 20 vectors, and with the first 30 vectors. Default is ``None``,
        which uses all vectors provided.
    metric : str, optional
        The metric used for calculating reconstruction error. Should be one of the options from
        ``scipy.spatial.distance.cdist``, or ``None`` if no scoring is required. Default is
        ``'correlation'``.
    checks : bool, optional
        Whether to verify types, shapes, and orthonormality of ``emodes`` and ``mass`` before
        reconstruction. Default is ``True``.
    **cdist_kwargs
        Additional keyword arguments to pass to ``scipy.spatial.distance.cdist``.

    Returns
    -------
    recon : numpy.ndarray
        The reconstructed data array of shape ``(n_verts, ..., n_recons)``, where ``n_recons`` is
        the number of different reconstructions ordered in ``mode_counts``. Each slice is the
        independent reconstruction of each map. Note that if ``mode_counts`` includes any constant
        vector (e.g., the first geometric eigenmode), the reconstructions will be constant for that
        value of ``mode_counts`` (this may also result in warnings/nans for ``recon_error``). 
    recon_error : numpy.ndarray
        The reconstruction error array of shape ``(..., n_recons)``. Each value represents the
        reconstruction error of one map. If ``metric`` is None, this will be empty. 
    beta : list of numpy.ndarray
        A list of beta coefficients calculated for each vector.
    
    Raises
    ------
    ValueError
        If ``mode_counts`` is not a 1D array-like of integers within the range [1, ``n_modes``].

    Notes
    -----
    If ``data`` contains NaNs or Infs, these will be disregarded during decomposition by masking
    corresponding vertices from ``data``, ``emodes``, and ``mass``. Note that this can lead to
    unexpected behaviour, such as extreme values in affected areas of the reconstructed data, or
    extreme beta values. This appears particularly prevalent when using the ``'regress'`` method.
    """
    # Format / validate arguments
    if checks is None:
        if method == 'regress':
            checks = 'shape'
        else:
            checks = True
    if checks is not False:
        ved = EigenData(emodes=emodes, mass=mass, data=data, checks=checks)
        emodes, mass, data = ved.emodes, ved.mass, ved.data

    # mode_counts is just shorthand for mode_ids
    # If mode_counts is provided, reformat into mode_ids
    mode_ids, squeeze_output = _process_mode_ids(mode_counts, mode_ids, emodes.shape[1])

    n_recons = len(mode_ids)

    beta = decompose(data, emodes, method=method, mass=mass, mode_ids=mode_ids, checks=False)
    
    # 4. Standardize shapes for math
    # data_2d: (n_verts, n_maps)
    data_2d = data.reshape(data.shape[0], -1)
    n_maps = data_2d.shape[1]
    
    # Prepare outputs
    # recon: (n_verts, n_maps, n_recons)
    recon_flat = np.empty((data_2d.shape[0], n_maps, n_recons))
    recon_error_flat = np.full((n_maps, n_recons), np.nan)

    for i in range(n_maps):
        # target_map: (1, n_verts)
        target_map = data_2d[:, [i]].T
        
        # Calculate all reconstructions for this map index 'i' across all mode_ids
        current_recons = []
        for j in range(n_recons):
            # 1. Force beta[j] to be 2D: (n_modes, n_maps)
            # 2. Slice the i-th column: (n_modes, 1)
            # 3. Multiply by emodes: (n_verts, n_modes) @ (n_modes, 1) -> (n_verts, 1)
            # b_j = np.atleast_2d(beta[j])

            
            # If beta was originally (n_modes,), atleast_2d makes it (1, n_modes)
            # We need it to be (n_modes, n_maps). 
            # Logic: If beta is (n_modes, n_maps), index [:, [i]] works.
            # If beta is (1, n_modes), we transpose and index [:, [0]].
            # if b_j.shape[0] == 1 and b_j.shape[1] == len(mode_ids[j]):
            #     b_j = b_j.T
                
            b_j = beta[j]
            b_j = b_j[:,np.newaxis] if b_j.ndim == 1 else b_j
            recon_j = emodes[:, mode_ids[j]] @ b_j[:, [i]]
            current_recons.append(recon_j.ravel()) # flatten to 1D for column_stack

        # map_recons: (n_verts, n_recons)
        map_recons = np.column_stack(current_recons)
        
        # Store for output
        recon_flat[:, i, :] = map_recons
        
        # cdist compare (1, n_verts) vs (n_recons, n_verts)
        recon_error_flat[i, :] = cdist(target_map, map_recons.T, metric=metric, **cdist_kwargs)

    # 6. Re-inflate shapes to original dimensions
    # recon: (n_verts, x, y, ..., n_recons)
    final_recon_shape = data.shape + (n_recons,)
    recon = recon_flat.reshape(final_recon_shape)
    
    # recon_error: (x, y, ..., n_recons)
    final_error_shape = data.shape[1:] + (n_recons,)
    recon_error = recon_error_flat.reshape(final_error_shape)

    if squeeze_output:
        recon = np.squeeze(recon, axis=-1)
        recon_error = np.squeeze(recon_error, axis=-1)
        beta = beta[0]

    return recon, recon_error, beta

def reconstruct_timeseries(
    timeseries: NDArray,
    emodes: NDArray,
    method: str = 'project',
    mass: csc_matrix | None = None,
    mode_counts: ArrayLike | None = None,
    mode_ids: ArrayLike | None = None,
    metric: _MetricCallback | _MetricKind | None = 'correlation',
    checks: bool | str = True,
    **cdist_kwargs
) -> Tuple[NDArray[floating], NDArray[floating], NDArray[floating], NDArray[floating],
           list[NDArray[floating]]]:
    """
    Calculate and score the reconstruction of the given timeseries data using the provided
    orthogonal vectors (e.g., geometric eigenmodes).

    Parameters
    ----------
    timeseries : array-like
        The input timeseries array of shape ``(n_verts, n_timepoints)``, where ``n_verts`` is the
        number of vertices and ``n_timepoints`` is the number of timepoints.
    emodes : array-like
        The vectors array of shape ``(n_verts, n_modes)``, where ``n_modes`` is the number of
        orthogonal vectors.
    method : str, optional
        The method used for the decomposition, either ``'project'`` to project data into a
        mass-orthonormal space or ``'regress'`` for least-squares fitting. Note that the beta values
        from ``'regress'`` tend towards those from ``'project'`` when more basis vectors are
        provided. For a non-orthonormal basis set, ``'regress'`` must be used. Default is
        ``'project'``.
    mass : array-like, optional
        The mass matrix of shape ``(n_verts, n_verts)`` used for the decomposition when method is
        ``'project'``. If vectors are orthonormal in Euclidean space, leave as ``None``. See
        :func:`eigen.is_orthonormal_basis` for more details. Default is ``None``.
    mode_counts : array-like, optional
        The sequence of vectors to be used for reconstruction. For example, ``mode_counts =
        np.array([10,20,30])`` will run three analyses: with the first 10 vectors, with the first 20
        vectors, and with the first 30 vectors. Default is ``None``, which uses all vectors
        provided.
    metric : str, optional
        The metric used for calculating reconstruction error. Should be one of the options from
        ``scipy.spatial.distance.cdist``, or ``None`` if no scoring is required. Default is
        ``'correlation'``.
    checks : bool, optional
        Whether to verify types, shapes, and orthonormality of ``emodes`` and ``mass`` before
        reconstruction. Default is ``True``.
    **cdist_kwargs
        Additional keyword arguments to pass to ``scipy.spatial.distance.cdist``.

    Returns
    -------
    fc_recon : numpy.ndarray
        The reconstructed functional connectivity array of shape ``(n_edges, n_recons)``, where
        ``n_edges = n_verts*(n_verts-1)/2`` and ``n_recons`` is the number of different
        reconstructions ordered in ``mode_counts``. The FC matrix returned is r-to-z (arctanh)
        transformed and vectorized. Note that if ``mode_counts`` includes any constant vector (e.g.,
        the first geometric eigenmode), the reconstructions will be constant for that value of
        ``mode_counts`` (this may also result in warnings/nans for ``recon_error``). 
    fc_recon_error : numpy.ndarray
        The functional reconstruction accuracy of shape ``(n_recons,)``. If ``metric`` is ``None``,
        this will be empty.
    recon : numpy.ndarray
        The reconstructed timeseries array of shape ``(n_verts, n_recons, n_timepoints)``, where
        ``n_recons`` is the number of different reconstructions ordered in ``mode_counts``. Each
        slice is the independent reconstruction of each timepoint. Note that if ``mode_counts``
        includes any constant vector (e.g., the first geometric eigenmode), the reconstructions will
        be constant for that value of ``mode_counts`` (this may also result in warnings/nans for
        ``recon_error``).
    recon_error : numpy.ndarray
        The reconstruction error array of shape ``(n_recons, n_timepoints)``. Each value represents
        the reconstruction error at one timepoint. If ``metric`` is ``None``, this will be empty. 
    beta : list of numpy.ndarray
        A list of beta coefficients calculated for each vector.
    
    Raises
    ------
    ValueError
        If ``timeseries`` does not have shape ``(n_verts, n_timepoints)``.
    """
    # Format / validate arguments
    if checks is not False:
        ved = EigenData(emodes=emodes, mass=mass, data=timeseries, checks=checks)
        emodes, mass, timeseries = ved.emodes, ved.mass, ved.data
        # timeseries = np.asarray_chkfinite(timeseries)
    if np.ndim(timeseries) != 2:
        raise ValueError("timeseries must have shape (n_verts, n_timepoints).")

    mode_ids, squeeze_output = _process_mode_ids(mode_counts, mode_ids, emodes.shape[1])
    
    # Use reconstruct to get independent reconstructions
    recon, recon_error, beta = reconstruct(
        timeseries,
        emodes, 
        method=method,
        mass=mass,
        mode_counts=mode_counts,
        metric=metric,
        checks=checks
    )

    # TODO : fix the relative sizings of inputs and outputs here
    if squeeze_output:
        recon = recon[:,:,np.newaxis]


    # Calculate FC of original timeseries
    fc = calc_vec_fc(timeseries)
    n_edges = len(fc)

    # Calculate FC of reconstructed timeseries
    n_recons = recon.shape[-1]
    fc_recon = np.empty((n_edges, n_recons))
    for i in range(n_recons):
        fc_recon[:, i] = calc_vec_fc(recon[..., i])

    # Score FC of reconstructions    
    fc_recon_error = cdist(
        fc_recon.T,
        fc[np.newaxis, :],
        metric=metric,
        **cdist_kwargs
        )[:, 0] if metric is not None else np.empty(n_recons)

    return fc_recon, fc_recon_error, recon, recon_error, beta

def calc_norm_power(
    beta: ArrayLike
) -> NDArray[floating]:
    """
    Transform beta coefficients from a decomposition into normalised power.

    Parameters
    ----------
    beta : array-like
        The beta coefficients array of shape ``(n_modes,)`` or ``(n_modes, n_maps)``, where
        ``n_modes`` is the number of orthogonal vectors and ``n_maps`` is the number of brain maps.

    Returns
    -------
    numpy.ndarray
        The normalized power array of shape ``(n_modes,)`` or ``(n_modes, n_maps)``, where each
        element represents the proportion of power contributed by the corresponding orthogonal
        vector to each brain map.
    """
    beta_sq = np.asarray_chkfinite(beta)**2
    total_power = np.sum(beta_sq, axis=0)

    return beta_sq / total_power

def calc_vec_fc(
    timeseries: NDArray
) -> NDArray[floating]:
    """
    Compute Fisher-z-transformed vectorized functional connectivity from timeseries data.
    
    Parameters
    ----------
    timeseries : array-like
        The input timeseries data of shape ``(n_verts, n_timepoints)``.

    Returns
    -------
    numpy.ndarray
        The Fisher-z-transformed vectorized functional connectivity array of shape ``(n_edges,)``,
        where ``n_edges = n_verts*(n_verts-1)/2``.
    """
    fc = np.corrcoef(timeseries)
    vec_fc = fc[np.triu_indices_from(fc, k=1)]
    return np.arctanh(vec_fc)

def _calc_beta(
    data: NDArray[floating],
    emodes: NDArray[floating],
    method: str,
    mass: csc_matrix | None,
    mask: NDArray
) -> NDArray[floating]:
    """Helper function to perform decomposition after validating arguments and masking NaNs/Infs."""
    d = data[mask, :]
    e = emodes[mask, :]
    if method == 'project' and mass is not None:
        m = mask_laplacian(stiffness=None, mass=mass, mask=mask)[1]
        return e.T @ m @ d
    elif method == 'project' and mass is None:
        return e.T @ d
    elif method == 'regress':
        return np.linalg.lstsq(e, d, rcond=None)[0]
    else:
        raise ValueError(f"Invalid method '{method}'; must be 'project' or 'regress'.")

def _process_mode_ids(mode_counts, mode_ids, n_modes): 
    # mode_counts is just shorthand for mode_ids
    # If mode_counts is provided, reformat into mode_ids
    squeeze_output = False
    if mode_ids is None and mode_counts is None:                # if both unspecified, generate list
        mode_ids = (np.arange(n_modes),)
        squeeze_output = True
    elif mode_counts is not None:                               # if counts provided, convert to list 
        if isinstance(mode_counts, int):
            mode_counts = [mode_counts]
            squeeze_output = True
        mode_ids = [np.arange(mc) for mc in mode_counts]
    if not isinstance(mode_ids, (list, tuple)):                 # check that ids are in a list/tuple
        raise ValueError("mode_ids must be a list or tuple of arrays of mode indices.")
    
    return mode_ids, squeeze_output