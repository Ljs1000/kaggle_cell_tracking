import geff
import networkx as nx
import numpy as np
import zarr

def load_tracks_array(geff_path: str) -> np.ndarray:
    """Load a geff tracks file and convert to napari's tracks format:
    columns = [track_id, t, z, y, x]
    """
    graph, metadata = geff.read(geff_path, backend="networkx")

    track_id_map = {}
    for i, component in enumerate(nx.weakly_connected_components(graph)):
        for node in component:
            track_id_map[node] = i

    tracks_data = np.array([
        [track_id_map[n], a["t"], a["z"], a["y"], a["x"]]
        for n, a in graph.nodes(data=True)
    ])
    return tracks_data

def load_image_array(zarr_path: str, mode: str = "r") -> zarr.Array:
    return zarr.open(zarr_path, mode=mode)["0"]