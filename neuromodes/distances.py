
import numpy as np
from scipy.sparse import coo_matrix
from lapy.diffgeo import tria_compute_gradient, tria_compute_divergence


def geodesic_heat(geometry, src_vids, splu_heat, splu_stiffness):

    shape = (geometry.v.shape[0],len(src_vids))
    # TODO : consider how to manage geodesic to any vertex vs geodesic to all vertices separately
    b0 = coo_matrix((np.ones_like(src_vids), (src_vids, np.arange(len(src_vids)))), shape=shape, dtype=np.float64).toarray()
                    
    vfunc = splu_heat.solve(b0)

    # Speeding this up would require changes to LaPy's tria_compute_gradient/_divergence to support parallel computation
    divf = np.zeros_like(b0)
    for i in range(len(src_vids)): 
        gradf = tria_compute_gradient(geometry, vfunc[:,i])  
        gradnorm = gradf / np.sqrt((gradf**2).sum(1))[:, np.newaxis]
        gradnorm = np.nan_to_num(gradnorm)
        divf[:,i] = tria_compute_divergence(geometry, gradnorm) # integrated divergence

    vf = splu_stiffness.solve(divf)
    vf -= np.min(vf, axis=0)

    return vf


