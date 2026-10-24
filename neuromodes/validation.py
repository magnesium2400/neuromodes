from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn
from collections.abc import Sequence

import numpy as np
from scipy.sparse import diags, spmatrix, csc_matrix

# from neuromodes.eigen import EigenData


if TYPE_CHECKING:
    from typing import Any, Literal, TypeAlias, TypeGuard

    from numpy.typing import NDArray
    from scipy.sparse import csc_matrix

    # _IntSequenceKind: TypeAlias = Sequence[int] | Sequence[np.integer]
    # _SeqSequenceKind: TypeAlias = Sequence[_IntSequenceKind] | NDArray[Any]

def _check_n_verts(
    n_verts: int | np.integer | None = None,
    **kwargs
) -> int | None:
    """Validate that all supplied matrices share the same spatial dimension (axis 0).

    Parameters
    ----------
    n_verts : int, np.integer, or None, optional
        Expected number of vertices. If provided, all matrices in kwargs must match this.
        If None, the spatial dimension is inferred from the first non-None matrix.
    **kwargs : array-like or None
        Named arrays/matrices (e.g. ``data_a=X, data_b=Y, mass=M``). Arguments set to
        ``None`` are automatically skipped.

    Returns
    -------
    int or None
        The inferred or validated number of vertices (``shape[0]``), or ``None`` if
        no non-None matrices were provided and ``n_verts`` was not specified.

    Raises
    ------
    TypeError
        If ``n_verts`` is a boolean/non-integer, or if an argument lacks a ``shape`` attribute.
    ValueError
        If ``n_verts <= 0``, or if any matrix's ``shape[0]`` fails to match ``n_verts``.
    """
    if n_verts is not None:
        if not _is_integer(n_verts):
            raise TypeError(
                f"n_verts must be an integer or None; got {n_verts} of type {type(n_verts).__name__}."
            )
        if n_verts <= 0:
            raise ValueError(f"n_verts must be positive; got {n_verts}.")

    target_nverts = n_verts

    for name, arr in kwargs.items():
        if arr is None:
            continue

        if not hasattr(arr, "shape") or len(arr.shape) == 0:
            raise TypeError(
                f"Argument '{name}' must be an array or matrix with a valid .shape attribute; "
                f"got type {type(arr).__name__}."
            )

        first_dim = arr.shape[0]

        if target_nverts is None:
            target_nverts = first_dim
        elif first_dim != target_nverts:
            msg = (
                f"Spatial dimension mismatch in argument '{name}': got shape[0] = {first_dim}, "
                f"but expected {target_nverts}"
            )
            if n_verts is None:
                msg += " (inferred from preceding inputs)."
            else:
                msg += " (specified by n_verts)."
            raise ValueError(msg)

    return int(target_nverts) if target_nverts is not None else None

# TODO P1 this function!
def _check_n_modes(
        n_modes_max = None, 
        mode_counts = None, 
        mode_ids = None,
        emodes = None, 
        evals = None,
):
    if emodes is not None and evals is not None:
        if emodes.shape[1] != evals.shape[0]: # TODO P1 change wording of this error and above (and tests)
            raise ValueError(f"must have shape Number of modes in emodes ({emodes.shape[1]}) does not match number of evals ({evals.shape[0]}).")
        n_modes_max = emodes.shape[1]
    # raise NotImplementedError("This function is not yet implemented. It will validate that the number of modes specified by mode_counts or mode_ids is consistent with the number of modes in emodes and evals.")


# check square
# check csc
def _check_mass(mass, 
                lump = None
                ):
    raise NotImplementedError("This function is not yet implemented. It will validate that the mass matrix is consistent with the number of vertices in emodes and evals.")
def _check_stiffness(stiffness,
                            ):
        raise NotImplementedError("This function is not yet implemented. It will validate that the stiffness matrix is consistent with the number of vertices in emodes and evals.")

def _check_emodes_evals(emodes,
                  evals,
                  mass, 
                  stiffness
                  ):
    # check emodes.T @ mass @ emodes == I
    # check emodes.T @ stiffness @ emodes == diag(evals)
    # check first emode (only) is constant
    # check first eval (only) is zero 
    raise NotImplementedError("This function is not yet implemented. It will validate that the eigenmodes are consistent with the mass and stiffness matrices and the eigenvalues.")



def _create_parcellation_matrix(
    parcellation: csc_matrix | NDArray[np.integer], 
    mask: NDArray[np.bool] | None = None
) -> csc_matrix:
    """
    Converts a parcellation vector to a sparse parcellation matrix.

    Parameters
    ----------
    parcellation : array-like
        The parcellation map, of shape ``(n_verts,)``, where each value is an integer representing
        the parcel ID for the corresponding vertex.

    Returns
    -------
    scipy.sparse.csc_matrix
        The sparse parcellation matrix, of shape ``(n_verts, n_parcels)``.

    Raises
    ------
    ValueError
        If ``parcellation`` is not 1D.
    """
    n_verts = parcellation.shape[0] # type: ignore
    mask = mask if mask is not None else np.ones(n_verts, dtype=bool)

    if parcellation.ndim == 1: # will be dense
        parc_mat = csc_matrix(
            (np.ones(np.sum(mask)),
            (np.arange(n_verts)[mask], parcellation[mask])),
            shape=(n_verts, np.max(parcellation)+1) # type: ignore
            )
    elif parcellation.ndim == 2:
        parc_mat = csc_matrix(parcellation[mask])
    else: 
        raise ValueError(f"Parcellation map must be 1D or 2D; got {parcellation.ndim}D.")

    return parc_mat


def _mass_to_areas(
    mass: csc_matrix | NDArray[np.floating] | None = None,
    n_verts: int | None = None,
    keepdims: bool = False
) -> NDArray[np.floating]:
    mass = np.asarray(_create_mass_matrix(mass=mass, n_verts=n_verts).sum(axis=1))
    if not keepdims:
        mass = mass.squeeze(axis=1)
    return mass

# TODO P3: consider adding dtype parameter to _create_mass_matrix for w=None case
# TODO: consider moving to EigenData
def _create_mass_matrix(
    mass: csc_matrix | NDArray[np.floating] | None = None,
    n_verts: int | None = None
) -> csc_matrix:
    
    if mass is None and n_verts is None:
        raise ValueError("Either mass or n_verts must be provided.")
    
    elif mass is None and n_verts is not None: # appease: pyright
        warn("Mass matrix not provided; assuming that area at each vertex is 1")
        output = diags(np.ones(n_verts), format='csc')

    elif isinstance(mass, spmatrix) and mass.shape is not None: # appease: pyright
        if mass.shape[0] == mass.shape[1]: 
            output = mass
        else: 
            raise ValueError(f"Sparse mass matrix has invalid shape: {mass.shape} (should be square).")

    elif isinstance(mass, np.ndarray) and mass.ndim == 2:
        if mass.shape[0] == mass.shape[1]:
            output = mass
        else: 
            raise ValueError(f"Dense mass matrix has invalid shape: {mass.shape} (should be square).")

    else: # 1D array, list, tuple, etc. 
        output = diags(np.ravel(np.asarray(mass)), format='csc')

    output = csc_matrix(output)
    
    if output.shape is None: 
        raise ValueError("Mass matrix has undefined shape.")
    elif output.shape[0] != output.shape[1]:
        raise ValueError(f"Mass matrix must be square; got shape {output.shape}.")
    if n_verts is not None and output.shape != (n_verts, n_verts):
        raise ValueError(f"Mass matrix has invalid shape: {output.shape} (should be ({n_verts}, "
                         f"{n_verts})).")

    return output




def _process_mode_ids(
    mode_counts: Sequence[int] | Sequence[np.integer] | int | np.integer | None,
    mode_ids: Sequence[Sequence[int]] | Sequence[Sequence[np.integer]] | None,
    n_modes_max: int | np.integer
) -> tuple[Sequence[Sequence[int] | Sequence[np.integer]], bool]:
    """
    Process mode_counts and mode_ids into a consistent format for use in decompose/reconstruct.
    """
    # mode_counts is just shorthand for mode_ids
    # If mode_counts is provided, reformat into mode_ids
    if mode_counts is not None and mode_ids is not None:
        raise UserWarning("Both mode_counts and mode_ids provided; mode_counts will be ignored.")
    
    if mode_ids is not None: # TODO add type checking here to ensure it is sensible
        if not _is_integer_sequence_of_sequences(mode_ids):
            raise TypeError(f"mode_ids must be an iterable of iterables of integers; got {type(mode_ids)}.")
        output = mode_ids
        squeeze_output = False
    elif mode_counts is None:
        output = [range(n_modes_max),]
        squeeze_output = True
    elif _is_integer(mode_counts):
        if mode_counts <= 0:
            raise ValueError(f"mode_counts must be a positive integer; got {mode_counts}.")
        if mode_counts > n_modes_max:
            raise ValueError(f"mode_counts must be in the range [1, n_modes = {n_modes_max}]; "
                             f"got {mode_counts}.")
        output = [range(mode_counts),]
        squeeze_output = True
    elif _is_integer_sequence(mode_counts):
        output = [range(mc) for mc in mode_counts]
        squeeze_output = False
    else: 
        raise TypeError(f"mode_counts must be a positive integer or an iterable of positive "
                        f"integers; got {type(mode_counts)}.")

    for i in output:
        if len(i) == 0:
            raise ValueError("mode_counts/mode_ids must not be empty. mode_counts must be a positive integer or sequence of positive integers.")
        for j in i: 
            if not isinstance(j, (int, np.integer)):
                raise TypeError(f"mode_counts/mode_ids must be integers; got {j} of type {type(j)}.")
            if j < 0 or j >= n_modes_max:
                raise ValueError(f"mode_counts/mode_ids must be in the range [0, n_modes = {n_modes_max}).")

    return output, squeeze_output


# TODO consider raising an error in these functions
# TODO consider returning the cleaned data of a specific type 
def _is_nonstring_sequence(x) -> TypeGuard[Sequence[Any]]:
    """
    Check if x is an iterable of numeric types (int or float).
    """
    return isinstance(x, Sequence) and not isinstance(x, (str, bytes))

def _is_integer(x) -> TypeGuard[int | np.integer]:
    """
    Check if x is an integer or an iterable of integers.
    """
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool) 

def _is_integer_sequence(x) -> TypeGuard[Sequence[int | np.integer]]:
    """
    Check if x is an iterable of integers.
    """
    return _is_nonstring_sequence(x) and all(_is_integer(i) for i in x)

def _is_integer_sequence_of_sequences(x) -> TypeGuard[Sequence[Sequence[int]] | Sequence[Sequence[np.integer]]]:
    """
    Check if x is an iterable of iterables of integers.
    """
    return _is_nonstring_sequence(x) and all(_is_integer_sequence(i) for i in x)
