"""
Utilities for loading Biohub cell-tracking data (GEFF tracks + Zarr images)
and building a PyTorch Dataset for cell detection training.
"""

from pathlib import Path
from typing import Any

import geff
import networkx as nx
import numpy as np
import zarr
from torch.utils.data import Dataset


def load_geff_graph(geff_path: str) -> tuple[nx.Graph, Any]:
    """Load a GEFF file as a NetworkX graph."""
    return geff.read(geff_path, backend="networkx")


def load_tracks_array(geff_path: str) -> np.ndarray:
    """
    Load a GEFF tracks file and convert it to a NumPy array.

    Each weakly connected component in the graph is treated as one track.

    Returns
    -------
    tracks : np.ndarray, shape (N_nodes, 5)
        Columns: [track_id, t, z, y, x]
    """
    graph, _metadata = load_geff_graph(geff_path)

    track_id_map = {
        node: track_id
        for track_id, component in enumerate(nx.weakly_connected_components(graph))
        for node in component
    }

    tracks = np.array(
        [
            [track_id_map[node], attrs["t"], attrs["z"], attrs["y"], attrs["x"]]
            for node, attrs in graph.nodes(data=True)
        ],
        dtype=np.float32,
    )

    return tracks


def load_image_array(zarr_path: str, mode: str = "r") -> zarr.Array:
    """
    Open the image data stored in a Zarr sample (lazily, no data is read yet).

    Expected shape: [T, Z, Y, X]
    """
    return zarr.open(zarr_path, mode=mode)["0"]


def get_centers_for_timepoint(tracks: np.ndarray, timepoint: int) -> np.ndarray:
    """
    Extract the cell centers belonging to one timepoint.

    Parameters
    ----------
    tracks : np.ndarray
        Array with columns [track_id, t, z, y, x].
    timepoint : int
        Timepoint to select.

    Returns
    -------
    centers : np.ndarray, shape (N_cells, 3)
        Columns: [z, y, x]
    """
    mask = tracks[:, 1] == timepoint
    return tracks[mask, 2:5].astype(np.float32)


class CellDataset(Dataset):

    """
    Dataset for training a 3D cell detection model.

    Each item represents one annotated timepoint from one sample.

    Returns:
        image:   3D image with shape [Z, Y, X]
        centers: Cell locations with shape [N, 3] as [z, y, x]
        metadata: Sample ID and timepoint

    Track annotations are loaded once, while image data is loaded
    only for the requested timepoint.
    """

    def __init__(self, train_dir):
        self.train_dir = Path(train_dir)

        zarr_paths = sorted(
            self.train_dir.glob("*.zarr")
        )

        self.volumes = {}
        self.tracks = {}
        self.samples = []

        for zarr_path in zarr_paths:
            sample_id = zarr_path.stem
            geff_path = self.train_dir / f"{sample_id}.geff"

            # Open Zarr once - still lazy, image data is not all loaded into RAM
            self.volumes[sample_id] = load_image_array(
                str(zarr_path)
            )

            # Load annotations once
            tracks = load_tracks_array(
                str(geff_path)
            )

            self.tracks[sample_id] = tracks

            annotated_timepoints = np.unique(
                tracks[:, 1]
            ).astype(int)

            for timepoint in annotated_timepoints:
                self.samples.append(
                    (sample_id, timepoint)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample_id, timepoint = self.samples[idx]

        volume = self.volumes[sample_id]
        tracks = self.tracks[sample_id]

        image = np.asarray(
            volume[timepoint],
            dtype=np.float32
        )

        centers = get_centers_for_timepoint(
            tracks,
            timepoint
        )

        metadata = {
            "sample_id": sample_id,
            "timepoint": timepoint
        }

        return image, centers, metadata


def construct_gaussian_heatmap(
    shape: tuple[int, int, int],
    centers: np.ndarray,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Create a 3D heatmap with a Gaussian peak at each cell center.

    Args:
        shape: Heatmap shape [Z, Y, X].
        centers: Cell centers with shape [N, 3] as [z, y, x].
        sigma: Gaussian standard deviation in voxels.

    Returns:
        3D heatmap with values in [0, 1].
    """
    heatmap = np.zeros(shape, dtype=np.float32)

    if len(centers) == 0:
        return heatmap

    radius = int(3 * sigma)

    for z, y, x in centers:
        z0 = max(0, int(z) - radius)
        z1 = min(shape[0], int(z) + radius + 1)

        y0 = max(0, int(y) - radius)
        y1 = min(shape[1], int(y) + radius + 1)

        x0 = max(0, int(x) - radius)
        x1 = min(shape[2], int(x) + radius + 1)

        zz, yy, xx = np.meshgrid(
            np.arange(z0, z1),
            np.arange(y0, y1),
            np.arange(x0, x1),
            indexing="ij",
        )

        squared_dist = (
            (zz - z) ** 2
            + (yy - y) ** 2
            + (xx - x) ** 2
        )

        gaussian = np.exp(
            -squared_dist / (2 * sigma**2)
        ).astype(np.float32)

        np.maximum(
            heatmap[z0:z1, y0:y1, x0:x1],
            gaussian,
            out=heatmap[z0:z1, y0:y1, x0:x1],
        )

    return heatmap