import struct
from pathlib import Path

import numpy as np


def _normal(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Compute unit normal for one triangle.

    Parameters:
        p0: First vertex ``[x, y, z]``.
        p1: Second vertex ``[x, y, z]``.
        p2: Third vertex ``[x, y, z]``.

    Returns:
        np.ndarray: Float32 unit normal; zero vector for degenerate triangle.
    """

    u = p1 - p0
    v = p2 - p0
    n = np.cross(u, v)
    norm = np.linalg.norm(n)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return (n / norm).astype(np.float32)


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

    buf = bytearray()
    header = b'dem_to_stl'.ljust(80, b' ')
    buf.extend(header)
    buf.extend(struct.pack('<I', len(triangles)))

    for p0, p1, p2 in triangles:
        n = _normal(p0, p1, p2)
        packed = struct.pack(
            '<12fH',
            n[0],
            n[1],
            n[2],
            p0[0],
            p0[1],
            p0[2],
            p1[0],
            p1[1],
            p1[2],
            p2[0],
            p2[1],
            p2[2],
            0,
        )
        buf.extend(packed)

    return bytes(buf)


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
