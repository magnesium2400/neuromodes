from __future__ import annotations
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs, eigsh

# def eigsw(A, M, **kwargs):
#     op = LinearOperator(
#         matvec=lambda x: M.dot(A.dot(M.dot(x))), # type: ignore
#         shape=M.shape,
#         dtype=M.dtype
#     )
#     return eigs(op, M=M, **kwargs)


def eigshw(A, M, **kwargs): 
    op = LinearOperator(
        matvec=lambda x: M.dot(A.dot(M.dot(x))), # type: ignore
        shape=M.shape,
        dtype=M.dtype
    )
    return eigsh(op, M=M, **kwargs)


def svdw_accurate(Y, M, n_components=10):
    """
    Geometry-aware SVD using the generalized symmetric eigsh solver.
    Bypasses non-symmetric eigs and matrix square roots entirely.
    """

    # To find SVD of Y, we want Y = U S Vt s.t. U.T M U = I
    # U are therefore the eigenmodes of Y Yt M
    # To find these eigenmodes, we solve the GEVP
    # M Y Yt M u = lambda M u

    Y = np.asarray(Y)
    
    # 1. Define the symmetric operator: M @ Y @ (Y.T @ (M @ x))
    # This matches the LHS matrix: (M @ Y @ Y.T @ M)
    YYt_operator = LinearOperator(
        matvec=lambda x: Y.dot(Y.T.dot(x)), # type: ignore
        shape=M.shape,
        dtype=Y.dtype
    )
    
    # 2. Solve the generalized symmetric problem using the wrapper utility
    eigenvalues, U = eigshw(
        A=YYt_operator, 
        M=M,
        k=n_components, 
        which='LM'
    )

    U = U[:, ::-1]  
    S = np.sqrt(eigenvalues[::-1])

    # 3. Project out the temporal modes: V.T = S^-1 @ U.T @ M @ Y
    Vt = U.T.dot(M.dot(Y)) / S[:, np.newaxis]
    
    return U, S, Vt


def svdw_fast(Y, M, n_components=10):
    """
    High-speed, truncated geometry-aware SVD for n_timepoints << n_vertices.
    Solves a subset of the temporal cross-product using sparse/iterative mechanics.
    """
    Y = np.asarray(Y)
    
    # 1. Compute the compact T x T temporal cross-product matrix
    temporal_cov = Y.T.dot(M.dot(Y))
    
    # 2. Extract only the top components from the temporal cross-product
    # Returns values in ascending algebraic order
    eigenvalues, V = eigsh(temporal_cov, k=n_components, which='LM')
    
    # Reverse matrices to match the standard descending SVD variance order
    V = V[:, ::-1]
    S = np.sqrt(eigenvalues[::-1])
    
    # 3. Project up to discover the spatial maps: U = Y @ V @ S^-1
    U = Y.dot(V) / S[np.newaxis, :]
        
    return U, S, V.T


def pcaw_accurate(Y, M, n_components=10):
    U, S, Vt = svdw_accurate(demeanw(Y, M), M, n_components=n_components)
    scores = S[:, np.newaxis] * Vt
    return U, scores, S


def pcaw_fast(Y, M, n_components=10):
    U, S, Vt = svdw_fast(demeanw(Y, M), M, n_components=n_components)
    scores = S[:, np.newaxis] * Vt
    return U, scores, S

