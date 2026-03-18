import numpy as np
from lapy import TriaMesh

def estimate_fwhm(geometry: TriaMesh, vfunc, mask=None):
    # Prelims
    rois = np.ones(len(geometry.v), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
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
     