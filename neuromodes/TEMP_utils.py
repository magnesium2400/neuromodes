"""Utility functions for mesh generation and visualization."""

import numpy as np
import tetgen
import plotly.graph_objs as go
from lapy import TetMesh
from matplotlib import colormaps

def make_thin_vol(surface_mesh, scaling=0.99, **kwargs):
    """
    Create a thin shell tetrahedral volume mesh from a surface mesh.
    
    Parameters
    ----------
    surface_mesh : trimesh.Trimesh
        The outer surface mesh.
    scaling : float, optional
        Scale factor for the inner surface (default: 0.99).
    **kwargs
        Additional keyword arguments for tetgen.TetGen.tetrahedralize().
    
    Returns
    -------
    tet_mesh : lapy.TetMesh
        Tetrahedral mesh of the shell volume.
    """
    # Create inner mesh
    inner_mesh = surface_mesh.copy()
    inner_mesh.apply_scale(scaling)
    
    # Combine vertices and faces
    vertices = np.vstack([surface_mesh.vertices, inner_mesh.vertices])
    outer_faces = surface_mesh.faces
    inner_faces = inner_mesh.faces + len(surface_mesh.vertices)
    faces = np.vstack([outer_faces, inner_faces])
    
    # Tetrahedralize
    tet = tetgen.TetGen(vertices, faces)
    tet.tetrahedralize(**kwargs)
    
    # Extract tetrahedral mesh
    vol_vertices = tet.grid.points
    vol_tets = tet.grid.cells.reshape(-1, 5)[:, 1:5].astype(int)
    
    return TetMesh(v=vol_vertices, t=vol_tets)

def plot_mesh_data(geometry, data, cmap='seismic_r', cnorm=True, width=700, height=700):
    """
    Plot a colored mesh surface with overlaid edges.
    
    Parameters
    ----------
    geometry : lapy.TriaMesh
        The surface mesh geometry.
    data : ndarray
        Data values (n_vertices,).
    cmap : str, optional
        Matplotlib colormap name (default: 'seismic_r').
    cnorm : bool, optional
        If data is 2D, whether to normalize color scale across frames (default: True).
    width : int, optional
        Figure width in pixels (default: 700).
    height : int, optional
        Figure height in pixels (default: 700).
    """
    # Make colormap for plotly
    cmap_obj = colormaps.get_cmap(cmap)
    vals = np.linspace(0, 1, 256)
    colorscale = [
        [i / (256 - 1), f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"]
        for i, (r, g, b, _) in enumerate(cmap_obj(vals))
    ]

    x, y, z = geometry.v.T
    i, j, k = geometry.t.T

    # Edge coordinates (static, shared across frames)
    edges = []
    for idx_i, idx_j, idx_k in zip(i, j, k):
        edges.append((idx_i, idx_j))
        edges.append((idx_j, idx_k))
        edges.append((idx_k, idx_i))
    edges = list(set(edges))

    edge_x, edge_y, edge_z = [], [], []
    for idx_i, idx_j in edges:
        edge_x.extend([x[idx_i], x[idx_j], None])
        edge_y.extend([y[idx_i], y[idx_j], None])
        edge_z.extend([z[idx_i], z[idx_j], None])

    is_animated = data.ndim == 2
    if is_animated:
        n_vertices, n_frames = data.shape
        if n_vertices != len(x):
            raise ValueError("data shape[0] must match number of vertices.")
        cmin, cmax = (np.nanmin(data), np.nanmax(data)) if cnorm else (np.nanmin(data[:, 0]), np.nanmax(data[:, 0]))

        # Base frame (first timepoint)
        base_mesh = go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            intensity=data[:, 0],
            colorscale=colorscale,
            cmin=cmin, cmax=cmax,
            flatshading=False,
            showscale=False,
        )


        frames = [
            go.Frame(
                data=[
                    go.Mesh3d(
                        x=x, y=y, z=z, i=i, j=j, k=k,
                        intensity=data[:, t],
                        colorscale=colorscale,
                        cmin=(cmin if cnorm else np.nanmin(data[:, t])), cmax=(cmax if cnorm else np.nanmax(data[:, t])),
                        flatshading=False,
                        showscale=False,
                    )
                ],
                name=str(t),
            )
            for t in range(n_frames)
        ]

        fig = go.Figure(data=[base_mesh], frames=frames)
    else:
        cmin, cmax = np.nanmin(data), np.nanmax(data)
        fig = go.Figure(data=[
            go.Mesh3d(
                x=x, y=y, z=z, i=i, j=j, k=k,
                intensity=data,
                colorscale=colorscale,
                cmin=cmin, cmax=cmax,
                flatshading=False,
                showscale=False,
            )
        ])

    # Add static edge overlay
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='black', width=1),
        showlegend=False,
        hoverinfo='skip',
    ))

    # Layout + optional animation controls
    layout_kwargs = dict(
        width=width,
        height=height,
        scene=dict(aspectmode="data"),
    )

    if is_animated:
        slider_steps = [
            dict(method="animate", args=[[str(t)], dict(mode="immediate", frame=dict(duration=0), transition=dict(duration=0))], label=str(t))
            for t in range(n_frames)
        ]
        sliders = [dict(active=0, pad=dict(t=50), steps=slider_steps)]
        play_pause = [
            dict(label="▶️ Play", method="animate",
                 args=[None, dict(frame=dict(duration=50, redraw=True),
                                  transition=dict(duration=0),
                                  fromcurrent=True, mode="immediate")]),
            dict(label="⏸️ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    transition=dict(duration=0),
                                    mode="immediate")]),
        ]
        layout_kwargs.update(
            sliders=sliders,
            updatemenus=[dict(type="buttons", showactive=False, buttons=play_pause, x=0, y=0)],
        )

    fig.update_layout(**layout_kwargs)
    fig.show()
