import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from lapy import Solver
from lapy.diffgeo import tria_compute_gradient, tria_compute_divergence

def geodesic_heat(geometry, src_vids, splu_heat, splu_stiffness):
    # TODO : 
    # 1. investigate timings on 32k mesh. 
    # 2. ensure this functionality has (future?) compatibility with use_cholmod results (i.e. should not be specific to splu) 
    # 3. investigate if modes can be used to improve splu_stiffness.solve -- in theory this can be approximated using a decomposition of divf

    shape = (geometry.v.shape[0], len(src_vids))
    # TODO : consider how to manage geodesic to any vertex vs geodesic to all vertices separately
    b0 = coo_matrix((np.ones_like(src_vids), (src_vids, np.arange(len(src_vids)))), shape=shape, dtype=np.float64).toarray()
                    
    vfunc = splu_heat.solve(b0)

    # LaPy now supports parallel computation of grads and divs (1.6.0+)
    gradf = tria_compute_gradient(geometry, vfunc)
    gradnorm = gradf / np.sqrt((gradf**2).sum(-1))[...,np.newaxis]
    gradnorm = np.nan_to_num(gradnorm)
    divf = tria_compute_divergence(geometry, gradnorm) # integrated divergence

    vf = splu_stiffness.solve(divf)
    vf -= np.min(vf, axis=0)

    return vf

def geodesic_setup(geometry, m = 1.0):
    a, b = Solver._fem_tria(geometry, lump=True)
    splu_stiffness = splu(a)
    t = m * geometry.avg_edge_length() ** 2
    hmat = b + t * a
    splu_heat = splu(hmat)
    return dict(splu_heat=splu_heat, splu_stiffness=splu_stiffness, geometry=geometry)
