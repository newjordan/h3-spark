"""Host-only ownership/layout model for the one-CTA H3 static prototype."""

from __future__ import annotations


WIDTH = 5376
VEC = 4
THREADS = 128
VECTOR_COUNT = WIDTH // VEC
VECTORS_PER_THREAD = (VECTOR_COUNT + THREADS - 1) // THREADS
FP4_BLOCK = 16
SCALE_COLS = WIDTH // FP4_BLOCK


def padded_rows(rows: int) -> int:
    if rows <= 0:
        raise ValueError("rows must be positive")
    return (rows + 15) // 16 * 16


def scale_rows(rows: int) -> int:
    return (padded_rows(rows) + 127) // 128 * 128


def owned_vectors(tid: int) -> tuple[int, ...]:
    if not 0 <= tid < THREADS:
        raise ValueError("tid must be in [0, 128)")
    return tuple(
        index
        for iteration in range(VECTORS_PER_THREAD)
        if (index := tid + iteration * THREADS) < VECTOR_COUNT
    )


def owned_columns(tid: int) -> tuple[int, ...]:
    return tuple(
        col
        for vector in owned_vectors(tid)
        for col in range(vector * VEC, vector * VEC + VEC)
    )


def scale_leaders() -> tuple[tuple[int, int, int], ...]:
    """Return (iteration, tid, scale_col) for every real scale write."""
    writes: list[tuple[int, int, int]] = []
    for iteration in range(VECTORS_PER_THREAD):
        for tid in range(0, THREADS, FP4_BLOCK // VEC):
            vector = tid + iteration * THREADS
            if vector < VECTOR_COUNT:
                writes.append((iteration, tid, vector * VEC // FP4_BLOCK))
    return tuple(writes)


def swizzled_scale_offset(row: int, col: int, col_length: int = SCALE_COLS) -> int:
    if row < 0 or not 0 <= col < col_length:
        raise ValueError("scale coordinate out of range")
    row_block = row // 128
    rem = row % 128
    d4 = rem // 32
    d3 = rem % 32
    col_block_group = col // 4
    d5 = col % 4
    groups = (col_length + 3) // 4
    return ((row_block * groups + col_block_group) * 32 + d3) * 16 + d4 * 4 + d5


def traffic_bytes(rows: int) -> dict[str, int]:
    """Lower-bound activation traffic, excluding tiny tables and output scales."""
    x_bytes = rows * WIDTH * 2
    q_bytes = padded_rows(rows) * WIDTH // 2
    return {
        # v2 first pass reads x for RMS and again for global amax; its second
        # pass reads x a third time.  It also writes rstd and packed output.
        "dynamic_v2": 3 * x_bytes + rows * 4 + q_bytes,
        # The static CTA reads x once into registers and directly writes FP4.
        "static_register": x_bytes + q_bytes,
        "saved": 2 * x_bytes + rows * 4,
    }


__all__ = [
    "FP4_BLOCK",
    "SCALE_COLS",
    "THREADS",
    "VEC",
    "VECTOR_COUNT",
    "VECTORS_PER_THREAD",
    "WIDTH",
    "owned_columns",
    "owned_vectors",
    "padded_rows",
    "scale_leaders",
    "scale_rows",
    "swizzled_scale_offset",
    "traffic_bytes",
]
