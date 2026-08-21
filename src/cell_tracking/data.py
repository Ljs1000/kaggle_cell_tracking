import geff
import networkx as nx
import numpy as np
import zarr

from pathlib import Path
from torch.utils.data import Dataset


#Class for selecting a specific sample from the dataset, use for training

class CellDataset(Dataset):

    def __init__(self, train_dir):
        self.train_dir = Path(train_dir)

        self.samples = sorted(
            self.train_dir.glob("*.zarr")
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        zarr_path = self.samples[idx]

        sample_id = zarr_path.stem
        geff_path = self.train_dir / f"{sample_id}.geff"

        volume = load_image_array(str(zarr_path))

        tracks = load_tracks_array(str(geff_path))

        return volume, tracks



def load_geff_graph(geff_path: str):
    return geff.read(geff_path, backend="networkx")


def load_tracks_array(geff_path: str) -> np.ndarray:
    """
    Load a GEFF tracks file and convert to napari's tracks format.

    Columns:
    [track_id, t, z, y, x]
    """
    graph, metadata = load_geff_graph(geff_path)

    track_id_map = {}

    for i, component in enumerate(nx.weakly_connected_components(graph)):
        for node in component:
            track_id_map[node] = i

    tracks_data = np.array([
        [
            track_id_map[n],
            a["t"],
            a["z"],
            a["y"],
            a["x"]
        ]
        for n, a in graph.nodes(data=True)
    ])

    return tracks_data


def load_image_array(zarr_path: str, mode: str = "r") -> zarr.Array:
    return zarr.open(zarr_path, mode=mode)["0"]


def get_centers_for_timepoint(
    tracks: np.ndarray,
    timepoint: int
) -> np.ndarray:
    """
    Return cell centers [z, y, x] for a single timepoint.
    """
    mask = tracks[:, 1] == timepoint

    return tracks[mask][:, 2:5]


def construct_gaussian_heatmap(
    shape,
    centers,
    sigma=2.0
):
    """
    TODO:
    Create a 3D heatmap with Gaussian peaks at every cell center.
    """
    heatmap = np.zeros(shape, dtype=np.float32)

    # implement Gaussian generation later

    return heatmap

