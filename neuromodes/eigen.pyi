from pathlib import Path
from typing import Any, overload, TypeAlias, Literal
from lapy import Solver, TriaMesh
from nibabel.gifti.gifti import GiftiImage
from numpy import floating, integer, bool_
from numpy.random import Generator
from numpy.typing import NDArray, ArrayLike
from scipy.sparse import csc_matrix

from neuromodes.basis import (_DecompositionKind, _IntSequenceKind, _SeqSequenceKind, _MetricCallbackKind, 
                              _ReconList, _ReconSingle, 
                              _ReconTSSingle, _ReconTSList)

_CheckKind: TypeAlias = bool | Literal['maps', 'ortho', 'shape', 'evals'] | None

# ==========================================
# CLASSES
# ==========================================

class EigenSolver(Solver):
    geometry: str | Path | GiftiImage | TriaMesh | dict  # Can be TriaMesh or similar internally
    n_verts: int
    mask: NDArray[bool_] | None
    hetero: NDArray[floating] | None
    use_cholmod: bool
    stiffness: csc_matrix
    mass: csc_matrix
    n_modes: int
    evals: NDArray[floating]
    emodes: NDArray[floating]
    _scaling: str | None
    _alpha: float | None

    def __init__(
        self,
        geometry: str | Path | GiftiImage | TriaMesh | dict,
        mask: ArrayLike | None = ...,
        normalize: bool = ...,
        hetero: ArrayLike | None = ...,
        alpha: float | None = ...,
        scaling: str | None = ...
    ) -> None: ...

    def __str__(self) -> str: ...

    def compute_lbo(self, lump: bool = ...) -> EigenSolver: ...

    def solve(
        self,
        n_modes: int,
        standardize: bool = ...,
        fix_mode1: bool = ...,
        lump: bool = ...,
        atol: float = ...,
        rtol: float = ...,
        sigma: float = ...,
        seed: int | Generator | None = ...,
        v0: ArrayLike | None = ...
    ) -> EigenSolver: ...

    def _check_for_emodes(self) -> None: ...

    # %% DECOMPOSE
    # 1. mode_counts is None or int -> Single Array 
    @overload
    def decompose(
        self,
        data: NDArray[floating],
        method: _DecompositionKind = ...,
        *,
        mode_counts: int | None = ...,
        mode_ids: None = ...,
        checks: _CheckKind = ...
    ) -> NDArray[floating]: ...

    # 2. mode_counts is Sequence -> List of Arrays
    @overload
    def decompose(
        self,
        data: NDArray[floating],
        method: _DecompositionKind = ...,
        *,
        mode_counts: _IntSequenceKind,
        mode_ids: None = ...,
        checks: _CheckKind = ...
    ) -> list[NDArray[floating]]: ...

    # 3. mode_ids is Sequence -> List of Arrays
    @overload
    def decompose(
        self,
        data: NDArray[floating],
        method: _DecompositionKind = ...,
        *,
        mode_counts: None = ...,
        mode_ids: _SeqSequenceKind,
        checks: _CheckKind = ...
    ) -> list[NDArray[floating]]: ...

    # %% RECONSTRUCT
    # 1. mode_counts is None or int -> Tuple with Single Array
    @overload
    def reconstruct(
        self,
        data: NDArray[floating],
        method: _DecompositionKind = ...,
        *,
        mode_counts: int | None = ...,
        mode_ids: None = ...,
        checks: _CheckKind = ...,
        metric: _MetricCallbackKind = ...,
        **cdist_kwargs
    ) -> _ReconSingle: ...

    # 2. mode_counts is Sequence -> Tuple with List of Arrays
    @overload
    def reconstruct(
        self,
        data: NDArray[floating],
        method: _DecompositionKind = ...,
        *,
        mode_counts: _IntSequenceKind,
        mode_ids: None = ...,
        checks: _CheckKind = ...,
        metric: _MetricCallbackKind = ...,
        **cdist_kwargs
    ) -> _ReconList: ...

    # 3. Mode IDs -> Tuple with List of Arrays
    @overload
    def reconstruct(
        self,
        data: NDArray[floating],
        method: _DecompositionKind = ...,
        *,
        mode_counts: None = ...,
        mode_ids: _SeqSequenceKind,
        checks: _CheckKind = ...,
        metric: _MetricCallbackKind = ...,
        **cdist_kwargs
    ) -> _ReconList: ...

    # %% RECONSTRUCT_TIMESERIES
    # 1. mode_counts is None or int -> Tuple with Single Array
    @overload
    def reconstruct_timeseries(
        self,
        timeseries: NDArray[floating],
        method: _DecompositionKind = ...,
        *,
        mode_counts: int | None = ...,
        mode_ids: None = ...,
        checks: _CheckKind = ...,
        metric: _MetricCallbackKind = ...,
        **cdist_kwargs
    ) -> _ReconTSSingle: ...

    # 2. mode_counts is Sequence -> Tuple with List of Arrays
    @overload
    def reconstruct_timeseries(
        self,
        timeseries: NDArray[floating],
        method: _DecompositionKind = ...,
        *,
        mode_counts: _IntSequenceKind,
        mode_ids: None = ...,
        checks: _CheckKind = ...,
        metric: _MetricCallbackKind = ...,
        **cdist_kwargs
    ) -> _ReconTSList: ...

    # 3. Mode IDs -> Tuple with List of Arrays
    @overload
    def reconstruct_timeseries(
        self,
        timeseries: NDArray[floating],
        method: _DecompositionKind = ...,
        *,
        mode_counts: None = ...,
        mode_ids: _SeqSequenceKind,
        checks: _CheckKind = ...,
        metric: _MetricCallbackKind = ...,
        **cdist_kwargs
    ) -> _ReconTSList: ...

    # --- OTHER WRAPPERS ---
    def compute_gem(self, **kwargs: Any) -> NDArray[floating]: ...
    def sim_nft_waves(self, **kwargs: Any) -> NDArray[floating]: ...
    def balloon_model(self, activity: ArrayLike, dt: float, **kwargs: Any) -> NDArray[floating]: ...
    def eigenstrap(self, data: NDArray[floating], **kwargs: Any) -> NDArray[floating]: ...


class EigenData:
    emodes: NDArray[floating]
    evals: NDArray[floating]
    mass: csc_matrix
    stiffness: csc_matrix
    scaled_hetero: NDArray[floating]
    data: NDArray[floating]

    def __init__(
        self,
        emodes: NDArray[floating] | None = ...,
        evals: NDArray[floating] | None = ...,
        mass: csc_matrix | None = ...,
        stiffness: csc_matrix | None = ...,
        scaled_hetero: NDArray[floating] | None = ...,
        data: NDArray[floating] | None = ...,
        checks: _CheckKind = ...
    ) -> None: ...

    def __getattribute__(self, name: str) -> Any: ...


# ==========================================
# FUNCTIONS
# ==========================================

def scale_hetero(
    hetero: ArrayLike,
    alpha: float = ...,
    scaling: Literal["exponential", "sigmoid"] = ...
) -> NDArray[floating]: ...

def standardize_emodes(
    emodes: NDArray[floating],
    checks: bool = ...
) -> NDArray[floating]: ...

def is_orthonormal_basis(
    emodes: NDArray[floating],
    mass: csc_matrix | None = ...,
    atol: float = ...,
    rtol: float = ...,
    checks: _CheckKind = ...
) -> bool: ...

def get_eigengroup_inds(
    n_modes: int
) -> list[NDArray[integer]]: ...