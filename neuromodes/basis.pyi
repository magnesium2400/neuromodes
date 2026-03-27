import numpy as np
from typing import overload, Union, List, Tuple, Any
from numpy.typing import NDArray
from scipy.sparse import csc_matrix

# 1. Defaults (Both None) -> Single Array
@overload
def decompose(
    data: NDArray,
    emodes: NDArray,
    method: str = ...,
    mass: Union[csc_matrix, None] = ...,
    *, 
    mode_counts: None = ...,
    mode_ids: None = ...,
    checks: str | bool = ...
) -> NDArray: ...

# 2. Int -> Single Array
@overload
def decompose(
    data: NDArray,
    emodes: NDArray,
    method: str = ...,
    mass: Union[csc_matrix, None] = ...,
    *, 
    mode_counts: int,
    mode_ids: None = ...,
    checks: str | bool = ...
) -> NDArray: ...

# 3. List/Tuple -> List of Arrays
@overload
def decompose(
    data: NDArray,
    emodes: NDArray,
    method: str = ...,
    mass: Union[csc_matrix, None] = ...,
    *, 
    mode_counts: List | Tuple,
    mode_ids: None = ...,
    checks: str | bool = ...
) -> List[NDArray]: ...

# 4. Mode IDs -> List of Arrays
@overload
def decompose(
    data: NDArray,
    emodes: NDArray,
    method: str = ...,
    mass: Union[csc_matrix, None] = ...,
    *, 
    mode_counts: None = ...,
    mode_ids: List | Tuple,
    checks: str | bool = ...
) -> List[NDArray]: ...





