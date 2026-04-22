import struct
from pathlib import Path

import numpy as np


def build_binary_stl_bytes(
        triangles: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> bytes:
    """Serialize triangles into binary STL payload.

    Parameters:
        triangles: List of vertex triplets.
            More triangles increase STL file size linearly.

    Returns:
        bytes: Complete binary STL bytes (80-byte header + facets).
    """

    triangle_count = len(triangles)
    header = b'dem_to_stl'.ljust(80, b' ')

    if triangle_count == 0:
        return header + struct.pack('<I', 0)

    tri_array = np.asarray(triangles, dtype=np.float32)
    p0 = tri_array[:, 0, :]
    p1 = tri_array[:, 1, :]
    p2 = tri_array[:, 2, :]

    # Compute all normals in one vectorized pass.
    u = p1 - p0
    v = p2 - p0
    normals = np.cross(u, v)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)

    normalized = np.zeros_like(normals, dtype=np.float32)
    valid = norms[:, 0] > 0.0
    if np.any(valid):
        normalized[valid] = normals[valid] / norms[valid]

    facet_dtype = np.dtype(
        [
            ('normal', '<f4', (3,)),
            ('p0', '<f4', (3,)),
            ('p1', '<f4', (3,)),
            ('p2', '<f4', (3,)),
            ('attr', '<u2'),
        ],
    )
    facets = np.zeros(triangle_count, dtype=facet_dtype)
    facets['normal'] = normalized
    facets['p0'] = p0
    facets['p1'] = p1
    facets['p2'] = p2

    return header + struct.pack('<I', triangle_count) + facets.tobytes()


def write_binary_stl(
        path: Path,
        triangles: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> int:
    """Write binary STL data to disk.

    Parameters:
        path: Output STL path.
        triangles: Triangle list to serialize.

    Returns:
        int: Number of triangles written.
    """

    data = build_binary_stl_bytes(triangles)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        f.write(data)
    return len(triangles)
